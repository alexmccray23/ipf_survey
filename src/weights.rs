use ipf::{ConvergenceConfig, ConvergenceReport, DenseMatrix, IpfError, IpfMatrix, cell_weights};

use crate::config::Normalization;
use crate::error::RakingError;
use crate::survey::CodedSurvey;
use crate::targets::TargetEntry;

// ---------------------------------------------------------------------------
// Stride helpers for N-D flat-data operations
// ---------------------------------------------------------------------------

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut strides = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Compute the 1-D marginal for `variable` (sum over all other dims).
fn compute_1d_marginal(
    data: &[f64],
    shape: &[usize],
    strides: &[usize],
    variable: usize,
) -> Vec<f64> {
    let levels = shape[variable];
    let stride = strides[variable];
    let mut marginal = vec![0.0; levels];
    for (i, &val) in data.iter().enumerate() {
        let coord = (i / stride) % levels;
        marginal[coord] += val;
    }
    marginal
}

/// Scale all cells along `variable` by per-level `factors`.
fn scale_variable(
    data: &mut [f64],
    shape: &[usize],
    strides: &[usize],
    variable: usize,
    factors: &[f64],
) {
    let levels = shape[variable];
    let stride = strides[variable];
    for (i, val) in data.iter_mut().enumerate() {
        let coord = (i / stride) % levels;
        *val *= factors[coord];
    }
}

// ---------------------------------------------------------------------------
// Custom IPF fitting loop (1-D marginal constraints for any N)
// ---------------------------------------------------------------------------

/// Run IPF with 1-D marginal constraints. Works for any number of variables.
///
/// The standard `ipf` crate solver uses complementary marginals (sum along one
/// axis, leaving product-of-other-dims entries). For N > 2 that doesn't map to
/// 1-D survey-raking constraints. This function implements the classical
/// Deming–Stephan iteration directly on flat data.
pub fn fit_marginals(
    matrix: &mut DenseMatrix<f64>,
    entries: &[TargetEntry],
    config: &ConvergenceConfig<f64>,
    diagnostics: bool,
) -> Result<ConvergenceReport<f64>, RakingError> {
    let shape = matrix.shape().to_vec();
    let strides = compute_strides(&shape);

    let mut residual_history = if diagnostics { Some(Vec::new()) } else { None };
    let tolerance: f64 = config.tolerance;

    for iter in 0..config.max_iterations {
        // Apply each marginal constraint in turn.
        {
            let data = matrix.flat_data_mut().unwrap();
            for entry in entries {
                let marginal =
                    compute_1d_marginal(data, &shape, &strides, entry.variable_index);
                let factors: Vec<f64> = entry
                    .targets
                    .iter()
                    .zip(marginal.iter())
                    .map(|(&t, &m)| if m.abs() > f64::EPSILON * 1e3 { t / m } else { 1.0 })
                    .collect();
                scale_variable(data, &shape, &strides, entry.variable_index, &factors);
            }
        }

        // Compute residual: max absolute difference between target and current marginals.
        let max_diff = max_marginal_diff(matrix.flat_data().unwrap(), &shape, &strides, entries);

        if let Some(ref mut hist) = residual_history {
            hist.push(max_diff);
        }

        if max_diff <= tolerance {
            return Ok(ConvergenceReport {
                converged: true,
                iterations: iter + 1,
                final_residual: max_diff,
                residual_history,
            });
        }
    }

    // Did not converge.
    let max_diff = max_marginal_diff(matrix.flat_data().unwrap(), &shape, &strides, entries);

    Err(IpfError::NotConverged(ConvergenceReport {
        converged: false,
        iterations: config.max_iterations,
        final_residual: max_diff,
        residual_history,
    })
    .into())
}

fn max_marginal_diff(
    data: &[f64],
    shape: &[usize],
    strides: &[usize],
    entries: &[TargetEntry],
) -> f64 {
    let mut max_diff = 0.0f64;
    for entry in entries {
        let marginal = compute_1d_marginal(data, shape, strides, entry.variable_index);
        for (&t, &m) in entry.targets.iter().zip(marginal.iter()) {
            max_diff = max_diff.max((t - m).abs());
        }
    }
    max_diff
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
    entries: &[TargetEntry],
    config: &ConvergenceConfig<f64>,
    diagnostics: bool,
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
        fit_marginals(fitted, entries, config, diagnostics)?;
    }

    Err(RakingError::TrimNotConverged {
        cycles: max_cycles,
    })
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
        let factors =
            DenseMatrix::from_shape_vec(vec![2, 2], vec![2.0, 2.0, 2.0, 2.0]).unwrap();
        let (weights, trimmed) = assign_weights(&survey, &factors, None);

        assert_eq!(weights, vec![2.0, 2.0, 2.0, 2.0]);
        assert_eq!(trimmed, 0);
    }

    #[test]
    fn assign_weights_with_bounds() {
        let survey = simple_survey();
        // factors: [0.1, 5.0, 1.0, 1.0] — first too low, second too high
        let factors =
            DenseMatrix::from_shape_vec(vec![2, 2], vec![0.1, 5.0, 1.0, 1.0]).unwrap();
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
    fn fit_marginals_2d() {
        // 2x2 uniform seed → fit to [3,3] x [3,3] (same grand total of 6)
        let mut matrix =
            DenseMatrix::from_shape_vec(vec![2, 2], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
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
        let report = fit_marginals(&mut matrix, &entries, &ConvergenceConfig::default(), false)
            .unwrap();
        assert!(report.converged);

        // All cells should be 1.5 (6 total / 4 cells)
        let data = matrix.flat_data().unwrap();
        for &v in data {
            assert!((v - 1.5).abs() < 1e-6, "cell = {v}, expected 1.5");
        }
    }

    #[test]
    fn fit_marginals_1d() {
        // 1D: 3 cells, fit to targets [4, 2, 4]
        let mut matrix =
            DenseMatrix::from_shape_vec(vec![3], vec![2.0, 2.0, 1.0]).unwrap();
        let entries = vec![TargetEntry {
            variable_index: 0,
            targets: vec![4.0, 2.0, 4.0],
        }];
        let report = fit_marginals(&mut matrix, &entries, &ConvergenceConfig::default(), false)
            .unwrap();
        assert!(report.converged);

        let data = matrix.flat_data().unwrap();
        assert!((data[0] - 4.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn fit_marginals_3d() {
        // 2x2x2 uniform seed, uniform targets → stays uniform
        let mut matrix =
            DenseMatrix::from_shape_vec(vec![2, 2, 2], vec![1.0; 8]).unwrap();
        let entries = vec![
            TargetEntry {
                variable_index: 0,
                targets: vec![4.0, 4.0],
            },
            TargetEntry {
                variable_index: 1,
                targets: vec![4.0, 4.0],
            },
            TargetEntry {
                variable_index: 2,
                targets: vec![4.0, 4.0],
            },
        ];
        let report = fit_marginals(&mut matrix, &entries, &ConvergenceConfig::default(), false)
            .unwrap();
        assert!(report.converged);

        let data = matrix.flat_data().unwrap();
        for &v in data {
            assert!((v - 1.0).abs() < 1e-6, "cell = {v}, expected 1.0");
        }
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

        let config = ConvergenceConfig::default();
        fit_marginals(&mut fitted, &entries, &config, false).unwrap();

        // With wide bounds, no trimming needed
        let report =
            trim_rerake(&seed, &mut fitted, (0.1, 100.0), &entries, &config, false, 50)
                .unwrap();
        assert_eq!(report.cycles, 0);
    }

    #[test]
    fn trim_not_converged() {
        let seed = DenseMatrix::from_shape_vec(vec![2, 2], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let mut fitted =
            DenseMatrix::from_shape_vec(vec![2, 2], vec![0.1, 5.0, 5.0, 0.1]).unwrap();

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

        let config = ConvergenceConfig::default();
        let result = trim_rerake(&seed, &mut fitted, (0.99, 1.01), &entries, &config, false, 0);
        assert!(matches!(
            result,
            Err(RakingError::TrimNotConverged { cycles: 0 })
        ));
    }

    #[test]
    fn strides_correct() {
        assert_eq!(compute_strides(&[3, 2]), vec![2, 1]);
        assert_eq!(compute_strides(&[3, 2, 4]), vec![8, 4, 1]);
        assert_eq!(compute_strides(&[5]), vec![1]);
    }

    #[test]
    fn marginal_1d_correct() {
        // shape [3, 2], data = [[1, 2], [3, 4], [5, 6]]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![3, 2];
        let strides = compute_strides(&shape);

        // Marginal of variable 0 (rows): [1+2, 3+4, 5+6] = [3, 7, 11]
        let m0 = compute_1d_marginal(&data, &shape, &strides, 0);
        assert_eq!(m0, vec![3.0, 7.0, 11.0]);

        // Marginal of variable 1 (cols): [1+3+5, 2+4+6] = [9, 12]
        let m1 = compute_1d_marginal(&data, &shape, &strides, 1);
        assert_eq!(m1, vec![9.0, 12.0]);
    }

    #[test]
    fn marginal_3d_correct() {
        // shape [2, 2, 2], all ones
        let data = vec![1.0; 8];
        let shape = vec![2, 2, 2];
        let strides = compute_strides(&shape);

        // Each variable should have marginal [4, 4] (each level sums to 4)
        for v in 0..3 {
            let m = compute_1d_marginal(&data, &shape, &strides, v);
            assert_eq!(m, vec![4.0, 4.0], "variable {v}");
        }
    }
}
