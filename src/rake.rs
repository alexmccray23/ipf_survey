use ipf::{IpfSolver, cell_weights};

use crate::config::RakingConfig;
use crate::diagnostics::{RakingDiagnostics, compute_diagnostics};
use crate::error::RakingError;
use crate::survey::CodedSurvey;
use crate::tabulate::tabulate;
use crate::targets::ValidatedTargets;
use crate::weights::{TrimReport, assign_weights, build_constraints, normalize, trim_rerake};

/// Result of a raking operation.
#[derive(Debug)]
pub struct RakingResult {
    /// Final per-record raking weights.
    pub weights: Vec<f64>,
    /// Convergence report from the IPF solver.
    pub convergence: ipf::ConvergenceReport<f64>,
    /// Weight distribution diagnostics (ESS, DEFF, summary stats).
    pub diagnostics: RakingDiagnostics,
}

/// Compute raking weights for a coded survey.
pub fn rake(
    survey: &CodedSurvey,
    targets: &ValidatedTargets,
    config: &RakingConfig,
) -> Result<RakingResult, RakingError> {
    // 1. Validate config
    config.validate()?;

    // 2. Cross-tabulate survey → seed matrix
    let seed = tabulate(survey);
    let mut fitted = seed.clone();

    // 3. Build solver and 1-D constraints
    let constraints = build_constraints(targets.entries())?;
    let solver = IpfSolver::<f64>::with_config(config.convergence.clone());

    // 4. Run IPF
    let mut convergence = solver.fit(&mut fitted, &constraints)?;

    // 5. Trim-rerake if bounds specified
    let trim_report;
    if let Some(bounds) = config.weight_bounds {
        let report = trim_rerake(
            &seed,
            &mut fitted,
            bounds,
            &solver,
            &constraints,
            config.max_trim_cycles,
        )?;
        trim_report = report;

        // Get convergence from the final state (after trim settled)
        let mut final_fitted = fitted.clone();
        convergence = solver.fit(&mut final_fitted, &constraints)?;
    } else {
        trim_report = TrimReport { cycles: 0 };
    }

    // 6. Compute adjustment factors and assign per-record weights
    let factors = cell_weights(&seed, &fitted)?;
    let (mut weights, trimmed_records) = assign_weights(survey, &factors, config.weight_bounds);

    let final_trimmed = if config.weight_bounds.is_some() {
        trimmed_records
    } else {
        0
    };

    // 7. Normalize
    normalize(&mut weights, &config.normalization, survey.n_records());

    // 8. Compute diagnostics
    let diagnostics = compute_diagnostics(&weights, trim_report.cycles, final_trimmed);

    Ok(RakingResult {
        weights,
        convergence,
        diagnostics,
    })
}

/// Rake with default config (no trimming, no normalization).
pub fn rake_simple(
    survey: &CodedSurvey,
    targets: &ValidatedTargets,
) -> Result<RakingResult, RakingError> {
    rake(survey, targets, &RakingConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Normalization;
    use crate::survey::Variable;
    use crate::targets::PopulationTargets;

    fn design_example() -> (CodedSurvey, ValidatedTargets) {
        let vars = vec![
            Variable {
                name: "age_group".into(),
                levels: 3,
                labels: None,
            },
            Variable {
                name: "gender".into(),
                levels: 2,
                labels: None,
            },
        ];

        let mut builder = CodedSurvey::builder(vars);
        builder.push_record(&[0, 0]).unwrap(); // young male
        builder.push_record(&[0, 1]).unwrap(); // young female
        builder.push_record(&[1, 0]).unwrap(); // middle male
        builder.push_record(&[1, 0]).unwrap(); // middle male
        builder.push_record(&[2, 1]).unwrap(); // senior female
        builder.push_record(&[2, 1]).unwrap(); // senior female
        let survey = builder.build().unwrap();

        let targets = PopulationTargets::new()
            .add("age_group", vec![2.0, 2.0, 2.0]) // age: equal
            .add("gender", vec![3.0, 3.0]) // gender: equal
            .validate(&survey)
            .unwrap();

        (survey, targets)
    }

    #[test]
    fn end_to_end_basic() {
        let (survey, targets) = design_example();
        let result = rake_simple(&survey, &targets).unwrap();

        assert_eq!(result.weights.len(), 6);
        // Weights should sum to grand total (6.0)
        let sum: f64 = result.weights.iter().sum();
        assert!(
            (sum - 6.0).abs() < 1e-6,
            "weights sum = {sum}, expected 6.0"
        );
    }

    #[test]
    fn weights_sum_to_grand_total() {
        let (survey, targets) = design_example();
        let result = rake(&survey, &targets, &RakingConfig::default()).unwrap();

        let sum: f64 = result.weights.iter().sum();
        assert!(
            (sum - 6.0).abs() < 1e-6,
            "weights sum = {sum}, expected 6.0"
        );
    }

    #[test]
    fn weighted_marginals_match_targets() {
        let (survey, targets) = design_example();
        let result = rake_simple(&survey, &targets).unwrap();

        // Check age marginals
        let mut age_marginals = [0.0; 3];
        for r in 0..survey.n_records() {
            let code = survey.record_codes(r)[0];
            age_marginals[code] += result.weights[r];
        }
        for (i, &m) in age_marginals.iter().enumerate() {
            assert!(
                (m - 2.0).abs() < 1e-6,
                "age marginal[{i}] = {m}, expected 2.0"
            );
        }

        // Check gender marginals
        let mut gender_marginals = [0.0; 2];
        for r in 0..survey.n_records() {
            let code = survey.record_codes(r)[1];
            gender_marginals[code] += result.weights[r];
        }
        for (i, &m) in gender_marginals.iter().enumerate() {
            assert!(
                (m - 3.0).abs() < 1e-6,
                "gender marginal[{i}] = {m}, expected 3.0"
            );
        }
    }

    #[test]
    fn trimming_keeps_factors_in_bounds() {
        let (survey, targets) = design_example();
        let config = RakingConfig {
            weight_bounds: Some((0.5, 3.0)),
            ..Default::default()
        };
        let result = rake(&survey, &targets, &config).unwrap();

        // Since base weights are all 1.0, the weight IS the factor
        for (i, &w) in result.weights.iter().enumerate() {
            assert!(
                w >= 0.5 - 1e-6 && w <= 3.0 + 1e-6,
                "weight[{i}] = {w} out of bounds [0.5, 3.0]"
            );
        }
    }

    #[test]
    fn normalization_sum_to_n() {
        let (survey, targets) = design_example();
        let config = RakingConfig {
            normalization: Normalization::SumToN,
            ..Default::default()
        };
        let result = rake(&survey, &targets, &config).unwrap();
        let sum: f64 = result.weights.iter().sum();
        assert!(
            (sum - 6.0).abs() < 1e-6,
            "with SumToN, sum = {sum}, expected 6.0"
        );
    }

    #[test]
    fn normalization_sum_to_value() {
        let (survey, targets) = design_example();
        let config = RakingConfig {
            normalization: Normalization::SumTo(100.0),
            ..Default::default()
        };
        let result = rake(&survey, &targets, &config).unwrap();
        let sum: f64 = result.weights.iter().sum();
        assert!(
            (sum - 100.0).abs() < 1e-6,
            "with SumTo(100), sum = {sum}, expected 100.0"
        );
    }

    #[test]
    fn diagnostics_invariants() {
        let (survey, targets) = design_example();
        let result = rake_simple(&survey, &targets).unwrap();

        let n = survey.n_records() as f64;
        assert!(
            result.diagnostics.effective_sample_size <= n + 1e-6,
            "ESS = {} > n = {}",
            result.diagnostics.effective_sample_size,
            n
        );
        assert!(
            result.diagnostics.design_effect >= 1.0 - 1e-6,
            "DEFF = {} < 1.0",
            result.diagnostics.design_effect
        );
    }

    #[test]
    fn identity_case() {
        // Targets that match the seed → all weights ~1.0
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
        // 4 records, one per cell
        let survey = CodedSurvey::from_flat_codes(vars, vec![0, 0, 0, 1, 1, 0, 1, 1], 4).unwrap();
        let targets = PopulationTargets::new()
            .add("a", vec![2.0, 2.0])
            .add("b", vec![2.0, 2.0])
            .validate(&survey)
            .unwrap();

        let result = rake_simple(&survey, &targets).unwrap();
        for (i, &w) in result.weights.iter().enumerate() {
            assert!((w - 1.0).abs() < 1e-6, "weight[{i}] = {w}, expected ~1.0");
        }
    }

    #[test]
    fn single_variable_raking() {
        let vars = vec![Variable {
            name: "x".into(),
            levels: 3,
            labels: None,
        }];
        // 5 records: 2 in cat 0, 2 in cat 1, 1 in cat 2
        let survey = CodedSurvey::from_flat_codes(vars, vec![0, 0, 1, 1, 2], 5).unwrap();
        let targets = PopulationTargets::new()
            .add("x", vec![4.0, 2.0, 4.0]) // want 4, 2, 4
            .validate(&survey)
            .unwrap();

        let result = rake_simple(&survey, &targets).unwrap();

        // cat 0: 2 records with factor 4/2 = 2.0
        assert!((result.weights[0] - 2.0).abs() < 1e-6);
        assert!((result.weights[1] - 2.0).abs() < 1e-6);
        // cat 1: 2 records with factor 2/2 = 1.0
        assert!((result.weights[2] - 1.0).abs() < 1e-6);
        assert!((result.weights[3] - 1.0).abs() < 1e-6);
        // cat 2: 1 record with factor 4/1 = 4.0
        assert!((result.weights[4] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn three_variable_raking() {
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
            Variable {
                name: "c".into(),
                levels: 2,
                labels: None,
            },
        ];
        // 8 records, one per cell of 2x2x2
        let codes: Vec<usize> = (0..8)
            .flat_map(|i| vec![i / 4, (i / 2) % 2, i % 2])
            .collect();
        let survey = CodedSurvey::from_flat_codes(vars, codes, 8).unwrap();

        let targets = PopulationTargets::new()
            .add("a", vec![4.0, 4.0])
            .add("b", vec![4.0, 4.0])
            .add("c", vec![4.0, 4.0])
            .validate(&survey)
            .unwrap();

        let result = rake_simple(&survey, &targets).unwrap();

        // Uniform targets on uniform seed → all weights ≈ 1.0
        for (i, &w) in result.weights.iter().enumerate() {
            assert!((w - 1.0).abs() < 1e-6, "weight[{i}] = {w}, expected ~1.0");
        }

        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 8.0).abs() < 1e-6);
    }

    #[test]
    fn convergence_report_populated() {
        let (survey, targets) = design_example();
        let result = rake_simple(&survey, &targets).unwrap();
        assert!(result.convergence.converged);
    }

    #[test]
    fn trimming_with_normalization() {
        let (survey, targets) = design_example();
        let config = RakingConfig {
            weight_bounds: Some((0.5, 3.0)),
            normalization: Normalization::SumTo(60.0),
            ..Default::default()
        };
        let result = rake(&survey, &targets, &config).unwrap();
        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 60.0).abs() < 1e-6, "sum = {sum}, expected 60.0");
    }
}
