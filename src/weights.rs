use ipf::{
    ConstraintSet, DenseMatrix, IpfMatrix, IpfSolver, ValidatedConstraints, cell_weights,
};

use crate::config::Normalization;
use crate::error::RakingError;
use crate::survey::CodedSurvey;
use crate::targets::TargetEntry;

// ---------------------------------------------------------------------------
// Constraint building helper
// ---------------------------------------------------------------------------

/// Build `ValidatedConstraints` from resolved target entries.
///
/// The entries have already been validated for grand-total consistency by
/// `PopulationTargets::validate`, so we use an infinite tolerance to skip
/// the redundant check in `ConstraintSet::build`.
pub fn build_constraints(
    entries: &[TargetEntry],
) -> Result<ValidatedConstraints<f64>, RakingError> {
    let mut cs = ConstraintSet::new();
    for entry in entries {
        cs = cs.add_1d(entry.variable_index, entry.targets.clone())?;
    }
    Ok(cs.build_with_tolerance(f64::INFINITY)?)
}

// ---------------------------------------------------------------------------
// Weight assignment
// ---------------------------------------------------------------------------

/// Maps cell adjustment factors to per-record weights.
///
/// When `bounds` is given, factors are clamped as a final safety net against
/// floating-point drift; after a successful [`trim_rerake`] all factors are
/// already within bounds.
pub fn assign_weights(
    survey: &CodedSurvey,
    factors: &DenseMatrix<f64>,
    bounds: Option<(f64, f64)>,
) -> Vec<f64> {
    let mut weights = Vec::with_capacity(survey.n_records());

    for r in 0..survey.n_records() {
        let codes = survey.record_codes(r);
        let mut factor = factors.get(codes);

        let bw = survey.base_weight(r);
        if let Some((lo, hi)) = bounds {
            // Skip structural-zero cells (factor 0 from 0/0): clamping would
            // fabricate weight for records that carry none.
            if bw > 0.0 && factor > 0.0 {
                factor = factor.clamp(lo, hi);
            }
        }

        weights.push(bw * factor);
    }

    weights
}

// ---------------------------------------------------------------------------
// Trim-rerake
// ---------------------------------------------------------------------------

/// Report from the trim-rerake loop.
pub struct TrimReport {
    /// Number of trim cycles run (0 = no factor needed trimming).
    pub cycles: usize,
    /// Flat per-cell mask: `true` for cells whose factor was frozen at a
    /// bound. Empty when trimming never ran.
    pub frozen: Vec<bool>,
}

/// Iterative freeze-and-rerake loop for weight trimming.
///
/// Each cycle clamps out-of-bounds adjustment factors at the cell level and
/// *freezes* those cells at `seed * bound`. The frozen mass is subtracted
/// from the marginal targets and IPF re-fits only the free cells, so a
/// binding bound cannot be undone by the re-rake (which is what made the
/// naive clamp-and-refit loop oscillate forever). Terminates because the
/// frozen set only grows.
///
/// On success the full matrix satisfies the original constraints and every
/// factor lies within `bounds`. Fails with [`RakingError::TrimInfeasible`]
/// when the frozen mass conflicts with a marginal target (the bounds are too
/// tight for the targets), or [`RakingError::TrimNotConverged`] when a
/// re-rake of the free cells cannot converge.
pub fn trim_rerake(
    seed: &DenseMatrix<f64>,
    fitted: &mut DenseMatrix<f64>,
    bounds: (f64, f64),
    solver: &IpfSolver<f64>,
    entries: &[TargetEntry],
    max_cycles: usize,
    tolerance: f64,
) -> Result<TrimReport, RakingError> {
    let shape = seed.shape().to_vec();
    let strides = crate::tabulate::strides(&shape);
    let seed_flat = seed.flat_data().unwrap().to_vec();
    let n_cells = seed_flat.len();
    let (lo, hi) = bounds;

    let mut frozen = vec![false; n_cells];
    let mut frozen_values = vec![0.0; n_cells];

    for cycle in 0..=max_cycles {
        // Find newly out-of-bounds factors among free, non-structural-zero
        // cells. Structural zeros (seed 0) have a meaningless 0/0 factor.
        let factors = cell_weights(seed, fitted)?;
        let factors_flat = factors.flat_data().unwrap();

        let mut any_trimmed = false;
        for i in 0..n_cells {
            if frozen[i] || seed_flat[i] == 0.0 {
                continue;
            }
            let factor = factors_flat[i];
            if factor < lo - tolerance {
                frozen[i] = true;
                frozen_values[i] = seed_flat[i] * lo;
                any_trimmed = true;
            } else if factor > hi + tolerance {
                frozen[i] = true;
                frozen_values[i] = seed_flat[i] * hi;
                any_trimmed = true;
            }
        }

        if !any_trimmed {
            return Ok(TrimReport {
                cycles: cycle,
                frozen,
            });
        }
        if cycle == max_cycles {
            break;
        }

        // Zero out frozen cells; IPF scaling keeps zero cells at zero, so
        // the fit below only moves the free cells.
        {
            let fitted_flat = fitted.flat_data_mut().unwrap();
            for i in 0..n_cells {
                if frozen[i] {
                    fitted_flat[i] = 0.0;
                }
            }
        }

        // Reduce each marginal target by the frozen mass in that margin
        // level, and check feasibility against the free mass that remains.
        let mut reduced_entries = Vec::with_capacity(entries.len());
        for entry in entries {
            let axis = entry.variable_index;
            let mut reduced = entry.targets.clone();
            let mut free_mass = vec![0.0; reduced.len()];

            for i in 0..n_cells {
                let level = (i / strides[axis]) % shape[axis];
                if frozen[i] {
                    reduced[level] -= frozen_values[i];
                } else {
                    free_mass[level] += seed_flat[i];
                }
            }

            for (level, r) in reduced.iter_mut().enumerate() {
                let eps = tolerance * entry.targets[level].abs().max(1.0);
                if *r < -eps || (*r > eps && free_mass[level] == 0.0) {
                    return Err(RakingError::TrimInfeasible {
                        variable: axis,
                        level,
                    });
                }
                *r = r.max(0.0);
            }

            reduced_entries.push(TargetEntry {
                variable_index: axis,
                targets: reduced,
            });
        }

        // Re-rake the free cells against the reduced targets.
        let reduced_constraints = build_constraints(&reduced_entries)?;
        let report = solver.fit(fitted, &reduced_constraints)?;
        if !report.converged {
            return Err(RakingError::TrimNotConverged { cycles: cycle + 1 });
        }

        // Restore the frozen cells.
        let fitted_flat = fitted.flat_data_mut().unwrap();
        for i in 0..n_cells {
            if frozen[i] {
                fitted_flat[i] = frozen_values[i];
            }
        }
    }

    Err(RakingError::TrimNotConverged { cycles: max_cycles })
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// Scale weights in place per the Normalization mode.
pub fn normalize(weights: &mut [f64], mode: &Normalization, n_records: usize) {
    let target = match mode {
        Normalization::None => return,
        Normalization::SumToN | Normalization::MeanOne => n_records as f64,
        Normalization::SumTo(v) => *v,
    };

    let sum: f64 = weights.iter().sum();
    if sum == 0.0 {
        return;
    }
    let scale = target / sum;
    for w in weights.iter_mut() {
        *w *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::survey::Variable;


    fn simple_survey() -> CodedSurvey {
        // 2x2: var0 has 2 levels, var1 has 2 levels
        // 4 records: (0,0), (0,1), (1,0), (1,1)
        let vars = vec![
            Variable {
                name: "a".into(),
                levels: 2,
                labels: None,
            },
            Variable {
                name: "b".into(),
                levels: 2,
                labels: None,
            },
        ];
        CodedSurvey::from_flat_codes(vars, vec![0, 0, 0, 1, 1, 0, 1, 1], 4).unwrap()
    }

    #[test]
    fn assign_weights_no_bounds() {
        let survey = simple_survey();
        let factors = DenseMatrix::from_shape_vec(vec![2, 2], vec![2.0, 2.0, 2.0, 2.0]).unwrap();
        let weights = assign_weights(&survey, &factors, None);

        assert_eq!(weights, vec![2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn assign_weights_with_bounds() {
        let survey = simple_survey();
        // factors: [0.1, 5.0, 1.0, 1.0] — first too low, second too high
        let factors = DenseMatrix::from_shape_vec(vec![2, 2], vec![0.1, 5.0, 1.0, 1.0]).unwrap();
        let weights = assign_weights(&survey, &factors, Some((0.5, 3.0)));

        assert_eq!(weights[0], 0.5); // clamped from 0.1 to 0.5
        assert_eq!(weights[1], 3.0); // clamped from 5.0 to 3.0
        assert_eq!(weights[2], 1.0);
        assert_eq!(weights[3], 1.0);
    }

    #[test]
    fn normalize_sum_to_n() {
        let mut weights = vec![2.0, 4.0, 6.0, 8.0];
        normalize(&mut weights, &Normalization::SumToN, 4);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 4.0).abs() < 1e-10);
    }

    #[test]
    fn normalize_sum_to_value() {
        let mut weights = vec![1.0, 2.0, 3.0];
        normalize(&mut weights, &Normalization::SumTo(100.0), 3);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 100.0).abs() < 1e-10);
    }

    #[test]
    fn normalize_mean_one() {
        let mut weights = vec![2.0, 4.0];
        normalize(&mut weights, &Normalization::MeanOne, 2);
        let mean = weights.iter().sum::<f64>() / 2.0;
        assert!((mean - 1.0).abs() < 1e-10);
    }

    #[test]
    fn normalize_none() {
        let mut weights = vec![2.0, 4.0];
        let original = weights.clone();
        normalize(&mut weights, &Normalization::None, 2);
        assert_eq!(weights, original);
    }

    #[test]
    fn trim_rerake_converges() {
        let seed = DenseMatrix::from_shape_vec(vec![2, 2], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let mut fitted = seed.clone();

        let entries = vec![
            TargetEntry {
                variable_index: 0,
                targets: vec![3.0, 3.0],
            },
            TargetEntry {
                variable_index: 1,
                targets: vec![3.0, 3.0],
            },
        ];

        let constraints = build_constraints(&entries).unwrap();
        let solver = IpfSolver::new();
        solver.fit(&mut fitted, &constraints).unwrap();

        // With wide bounds, no trimming needed
        let report = trim_rerake(&seed, &mut fitted, (0.1, 100.0), &solver, &entries, 50, 1e-6)
            .unwrap();
        assert_eq!(report.cycles, 0);
        assert!(report.frozen.iter().all(|&f| !f));
    }

    #[test]
    fn trim_binding_bounds_converge() {
        // Seed [[4,1],[1,1]], uniform margins 3.5/3.5. Unbounded factors are
        // {0.583, 1.167, 1.167, 2.333}; bounds (0.5, 2.0) bind cell (1,1).
        // The feasible solution freezes (1,1) at 2.0 and re-rakes the rest
        // to [[2.0, 1.5], [1.5, 2.0]].
        let seed = DenseMatrix::from_shape_vec(vec![2, 2], vec![4.0, 1.0, 1.0, 1.0]).unwrap();
        let mut fitted = seed.clone();

        let entries = vec![
            TargetEntry {
                variable_index: 0,
                targets: vec![3.5, 3.5],
            },
            TargetEntry {
                variable_index: 1,
                targets: vec![3.5, 3.5],
            },
        ];

        let constraints = build_constraints(&entries).unwrap();
        let solver = IpfSolver::new();
        solver.fit(&mut fitted, &constraints).unwrap();

        let report = trim_rerake(&seed, &mut fitted, (0.5, 2.0), &solver, &entries, 50, 1e-6)
            .unwrap();
        assert!(report.cycles > 0);
        assert_eq!(report.frozen, vec![false, false, false, true]);

        // Factors within bounds
        let factors = cell_weights(&seed, &fitted).unwrap();
        for &f in factors.flat_data().unwrap() {
            assert!(
                (0.5 - 1e-6..=2.0 + 1e-6).contains(&f),
                "factor {f} out of bounds"
            );
        }

        // Original margins still satisfied
        assert!((fitted.get(&[0, 0]) + fitted.get(&[0, 1]) - 3.5).abs() < 1e-8);
        assert!((fitted.get(&[0, 0]) + fitted.get(&[1, 0]) - 3.5).abs() < 1e-8);
        assert!((fitted.get(&[0, 0]) - 2.0).abs() < 1e-8);
    }

    #[test]
    fn trim_infeasible_one_dimension() {
        // 1-D: seed margin [2,2,1], target [4,2,4]. Level 2 needs factor 4.0;
        // with an upper bound of 2.0 the target can never be met.
        let seed = DenseMatrix::from_shape_vec(vec![3], vec![2.0, 2.0, 1.0]).unwrap();
        let mut fitted = seed.clone();

        let entries = vec![TargetEntry {
            variable_index: 0,
            targets: vec![4.0, 2.0, 4.0],
        }];

        let constraints = build_constraints(&entries).unwrap();
        let solver = IpfSolver::new();
        solver.fit(&mut fitted, &constraints).unwrap();

        let result = trim_rerake(&seed, &mut fitted, (0.5, 2.0), &solver, &entries, 50, 1e-6);
        assert!(matches!(
            result,
            Err(RakingError::TrimInfeasible {
                variable: 0,
                level: 2,
            })
        ));
    }

    #[test]
    fn trim_zero_budget() {
        let seed = DenseMatrix::from_shape_vec(vec![2, 2], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let mut fitted = DenseMatrix::from_shape_vec(vec![2, 2], vec![0.1, 5.0, 5.0, 0.1]).unwrap();

        let entries = vec![
            TargetEntry {
                variable_index: 0,
                targets: vec![5.1, 5.1],
            },
            TargetEntry {
                variable_index: 1,
                targets: vec![5.1, 5.1],
            },
        ];

        let solver = IpfSolver::new();
        let result = trim_rerake(&seed, &mut fitted, (0.99, 1.01), &solver, &entries, 0, 1e-6);
        assert!(matches!(
            result,
            Err(RakingError::TrimNotConverged { cycles: 0 })
        ));
    }
}
