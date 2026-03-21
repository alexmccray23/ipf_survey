/// Summary statistics for the weight distribution.
#[derive(Debug, Clone)]
pub struct WeightSummary {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    /// Coefficient of variation (`std_dev` / mean).
    pub cv: f64,
}

/// Full raking diagnostics.
#[derive(Debug, Clone)]
pub struct RakingDiagnostics {
    /// Effective sample size: ESS = (sum w)^2 / sum(w^2).
    pub effective_sample_size: f64,
    /// Design effect: DEFF = n / ESS.
    pub design_effect: f64,
    pub weight_summary: WeightSummary,
    pub trim_cycles: usize,
    pub trimmed_records: usize,
}

/// Compute diagnostics from a weights vector.
pub fn compute_diagnostics(
    weights: &[f64],
    trim_cycles: usize,
    trimmed_records: usize,
) -> RakingDiagnostics {
    let n = weights.len() as f64;

    // Pass 1: sum, sum_sq, min, max
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;

    for &w in weights {
        sum += w;
        sum_sq += w * w;
        if w < min {
            min = w;
        }
        if w > max {
            max = w;
        }
    }

    let mean = sum / n;

    // Pass 2: variance
    let mut var_sum = 0.0;
    for &w in weights {
        let diff = w - mean;
        var_sum += diff * diff;
    }
    let std_dev = (var_sum / n).sqrt();
    let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };

    // Median
    let median = {
        let mut sorted = weights.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = sorted.len();
        if len % 2 == 1 {
            sorted[len / 2]
        } else {
            f64::midpoint(sorted[len / 2 - 1], sorted[len / 2])
        }
    };

    // ESS = (sum w)^2 / sum(w^2)
    let ess = if sum_sq > 0.0 {
        (sum * sum) / sum_sq
    } else {
        0.0
    };

    // DEFF = n / ESS
    let deff = if ess > 0.0 { n / ess } else { f64::INFINITY };

    RakingDiagnostics {
        effective_sample_size: ess,
        design_effect: deff,
        weight_summary: WeightSummary {
            min,
            max,
            mean,
            median,
            cv,
        },
        trim_cycles,
        trimmed_records,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_weights() {
        let weights = vec![1.0; 10];
        let diag = compute_diagnostics(&weights, 0, 0);

        assert!((diag.effective_sample_size - 10.0).abs() < 1e-10);
        assert!((diag.design_effect - 1.0).abs() < 1e-10);
        assert!((diag.weight_summary.min - 1.0).abs() < 1e-10);
        assert!((diag.weight_summary.max - 1.0).abs() < 1e-10);
        assert!((diag.weight_summary.mean - 1.0).abs() < 1e-10);
        assert!((diag.weight_summary.median - 1.0).abs() < 1e-10);
        assert!((diag.weight_summary.cv).abs() < 1e-10);
    }

    #[test]
    fn known_distribution() {
        // weights: [1, 2, 3, 4]
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let diag = compute_diagnostics(&weights, 2, 1);

        let sum = 10.0;
        let sum_sq = 1.0 + 4.0 + 9.0 + 16.0; // 30
        let expected_ess = sum * sum / sum_sq; // 100/30 ≈ 3.333
        let expected_deff = 4.0 / expected_ess;

        assert!((diag.effective_sample_size - expected_ess).abs() < 1e-10);
        assert!((diag.design_effect - expected_deff).abs() < 1e-10);
        assert!((diag.weight_summary.min - 1.0).abs() < 1e-10);
        assert!((diag.weight_summary.max - 4.0).abs() < 1e-10);
        assert!((diag.weight_summary.mean - 2.5).abs() < 1e-10);
        assert_eq!(diag.trim_cycles, 2);
        assert_eq!(diag.trimmed_records, 1);
    }

    #[test]
    fn median_odd() {
        let weights = vec![3.0, 1.0, 2.0];
        let diag = compute_diagnostics(&weights, 0, 0);
        assert!((diag.weight_summary.median - 2.0).abs() < 1e-10);
    }

    #[test]
    fn median_even() {
        let weights = vec![4.0, 1.0, 3.0, 2.0];
        let diag = compute_diagnostics(&weights, 0, 0);
        // sorted: [1, 2, 3, 4], median = (2+3)/2 = 2.5
        assert!((diag.weight_summary.median - 2.5).abs() < 1e-10);
    }

    #[test]
    fn single_record() {
        let weights = vec![5.0];
        let diag = compute_diagnostics(&weights, 0, 0);

        assert!((diag.effective_sample_size - 1.0).abs() < 1e-10);
        assert!((diag.design_effect - 1.0).abs() < 1e-10);
        assert!((diag.weight_summary.median - 5.0).abs() < 1e-10);
    }

    #[test]
    fn cv_calculation() {
        // weights: [1, 3], mean = 2, std_dev = sqrt(((1-2)^2 + (3-2)^2)/2) = 1
        // cv = 1/2 = 0.5
        let weights = vec![1.0, 3.0];
        let diag = compute_diagnostics(&weights, 0, 0);
        assert!((diag.weight_summary.cv - 0.5).abs() < 1e-10);
    }
}
