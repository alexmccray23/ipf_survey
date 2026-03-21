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
/// Returns `(weights, trimmed_count)` where `trimmed_count` is the number of
/// records whose factor was clamped to a bound.
pub fn assign_weights(
    survey: &CodedSurvey,
    factors: &DenseMatrix<f64>,
    bounds: Option<(f64, f64)>,
) -> (Vec<f64>, usize) {
    let mut weights = Vec::with_capacity(survey.n_records());
    let mut trimmed = 0;

    for r in 0..survey.n_records() {
        let codes = survey.record_codes(r);
        let mut factor = factors.get(codes);

        let bw = survey.base_weight(r);
        if let Some((lo, hi)) = bounds {
            // Don't count structural-zero cells (factor 0 from 0/0) as trimmed
            if bw > 0.0 && (factor < lo || factor > hi) {
                trimmed += 1;
                factor = factor.clamp(lo, hi);
            }
        }

        weights.push(bw * factor);
    }

    (weights, trimmed)
}

// ---------------------------------------------------------------------------
// Trim-rerake
// ---------------------------------------------------------------------------

/// Report from the trim-rerake loop.
pub struct TrimReport {
    pub cycles: usize,
}

/// Iterative clamp-and-refit loop for weight trimming.
///
/// Clamps adjustment factors at the cell level, then re-runs IPF until stable.
pub fn trim_rerake(
    seed: &DenseMatrix<f64>,
    fitted: &mut DenseMatrix<f64>,
    bounds: (f64, f64),
    solver: &IpfSolver<f64>,
    constraints: &ValidatedConstraints<f64>,
    max_cycles: usize,
) -> Result<TrimReport, RakingError> {
    let seed_flat = seed.flat_data().unwrap();

    for cycle in 0..max_cycles {
        let factors = cell_weights(seed, fitted)?;
        let factors_flat = factors.flat_data().unwrap();
        let fitted_flat = fitted.flat_data_mut().unwrap();

        let mut any_trimmed = false;
        for (i, &factor) in factors_flat.iter().enumerate() {
            // Skip structural zeros: if seed cell is zero, both seed and fitted
            // are zero and the factor is meaningless (0/0 → 0). Clamping would
            // set fitted = 0 * bound = 0, which changes nothing but prevents
            // convergence.
            if seed_flat[i] == 0.0 {
                continue;
            }
            if factor < bounds.0 {
                fitted_flat[i] = seed_flat[i] * bounds.0;
                any_trimmed = true;
            } else if factor > bounds.1 {
                fitted_flat[i] = seed_flat[i] * bounds.1;
                any_trimmed = true;
            }
        }

        if !any_trimmed {
            return Ok(TrimReport { cycles: cycle });
        }

        // Re-rake with the clamped matrix.
        solver.fit(fitted, constraints)?;
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
        let (weights, trimmed) = assign_weights(&survey, &factors, None);

        assert_eq!(weights, vec![2.0, 2.0, 2.0, 2.0]);
        assert_eq!(trimmed, 0);
    }

    #[test]
    fn assign_weights_with_bounds() {
        let survey = simple_survey();
        // factors: [0.1, 5.0, 1.0, 1.0] — first too low, second too high
        let factors = DenseMatrix::from_shape_vec(vec![2, 2], vec![0.1, 5.0, 1.0, 1.0]).unwrap();
        let (weights, trimmed) = assign_weights(&survey, &factors, Some((0.5, 3.0)));

        assert_eq!(weights[0], 0.5); // clamped from 0.1 to 0.5
        assert_eq!(weights[1], 3.0); // clamped from 5.0 to 3.0
        assert_eq!(weights[2], 1.0);
        assert_eq!(weights[3], 1.0);
        assert_eq!(trimmed, 2);
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
        let report = trim_rerake(&seed, &mut fitted, (0.1, 100.0), &solver, &constraints, 50)
            .unwrap();
        assert_eq!(report.cycles, 0);
    }

    #[test]
    fn trim_not_converged() {
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

        let constraints = build_constraints(&entries).unwrap();
        let solver = IpfSolver::new();
        let result = trim_rerake(&seed, &mut fitted, (0.99, 1.01), &solver, &constraints, 0);
        assert!(matches!(
            result,
            Err(RakingError::TrimNotConverged { cycles: 0 })
        ));
    }
}
