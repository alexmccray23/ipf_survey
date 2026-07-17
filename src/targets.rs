use crate::RakingError;
use crate::survey::CodedSurvey;

/// A validated marginal-target entry: resolved variable index and target values.
#[derive(Debug, Clone)]
pub struct TargetEntry {
    pub variable_index: usize,
    pub targets: Vec<f64>,
}

/// An unvalidated entry storing the variable name as specified by the caller.
#[derive(Debug, Clone)]
struct UnvalidatedEntry {
    variable_name: String,
    targets: Vec<f64>,
}

/// Builder for assembling population targets before validation.
#[derive(Debug, Clone)]
pub struct PopulationTargets {
    entries: Vec<UnvalidatedEntry>,
}

impl PopulationTargets {
    /// Create an empty target set.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add marginal targets for a variable by name.
    ///
    /// The name must match one of the `Variable::name` fields in the survey
    /// passed to [`validate`](Self::validate). Target vector length must equal
    /// that variable's `levels` count.
    #[must_use]
    pub fn add(mut self, variable_name: &str, targets: Vec<f64>) -> Self {
        self.entries.push(UnvalidatedEntry {
            variable_name: variable_name.to_owned(),
            targets,
        });
        self
    }

    /// Validate against a survey with the default grand-total tolerance
    /// (1e-6, relative — see [`validate_with_tolerance`](Self::validate_with_tolerance)).
    pub fn validate(self, survey: &CodedSurvey) -> Result<ValidatedTargets, RakingError> {
        self.validate_with_tolerance(survey, 1e-6)
    }

    /// Validate against a survey with a custom grand-total tolerance.
    ///
    /// `tolerance` is interpreted **relative** to the grand total `G`
    /// (clamped to at least 1), matching the `ipf` crate's convention:
    /// grand totals are consistent when `|G - total_i| <= tolerance * max(G, 1)`.
    /// This keeps the default meaningful whether targets sum to 30 or to
    /// hundreds of millions — population-scale margins that were rounded to
    /// whole people still validate, while genuinely mismatched totals fail.
    pub fn validate_with_tolerance(
        self,
        survey: &CodedSurvey,
        tolerance: f64,
    ) -> Result<ValidatedTargets, RakingError> {
        if self.entries.is_empty() {
            return Err(RakingError::NoTargets);
        }

        // Check for duplicate variable names
        for (i, a) in self.entries.iter().enumerate() {
            for b in &self.entries[i + 1..] {
                if a.variable_name == b.variable_name {
                    return Err(RakingError::DuplicateVariable {
                        name: a.variable_name.clone(),
                    });
                }
            }
        }

        // Resolve names to indices and validate
        let mut resolved = Vec::with_capacity(self.entries.len());
        let mut names = Vec::with_capacity(self.entries.len());
        for entry in self.entries {
            let Some(var_index) = survey
                .variables()
                .iter()
                .position(|v| v.name == entry.variable_name)
            else {
                return Err(RakingError::VariableNotFound {
                    name: entry.variable_name,
                });
            };

            let expected = survey.variables()[var_index].levels;
            if entry.targets.len() != expected {
                return Err(RakingError::TargetLengthMismatch {
                    name: entry.variable_name,
                    expected,
                    got: entry.targets.len(),
                });
            }

            for (i, &t) in entry.targets.iter().enumerate() {
                if !t.is_finite() || t < 0.0 {
                    return Err(RakingError::InvalidTarget {
                        name: entry.variable_name,
                        index: i,
                        value: t,
                    });
                }
            }

            names.push(entry.variable_name);
            resolved.push(TargetEntry {
                variable_index: var_index,
                targets: entry.targets,
            });
        }

        // Compute grand totals and check consistency
        let totals: Vec<f64> = resolved.iter().map(|e| e.targets.iter().sum()).collect();

        let grand_total = totals[0];
        // Relative tolerance, scaled by the grand total (clamped to at
        // least 1 so zero/small totals stay on an absolute scale).
        let threshold = tolerance * grand_total.max(1.0);
        for (i, &total) in totals.iter().enumerate().skip(1) {
            let diff = (grand_total - total).abs();
            if diff > threshold {
                return Err(RakingError::InconsistentTotals {
                    variable_a: names[0].clone(),
                    variable_b: names[i].clone(),
                    diff,
                });
            }
        }

        Ok(ValidatedTargets {
            entries: resolved,
            grand_total,
        })
    }
}

impl Default for PopulationTargets {
    fn default() -> Self {
        Self::new()
    }
}

/// Validated, sealed population targets. Only constructable via [`PopulationTargets::validate`].
#[derive(Debug)]
pub struct ValidatedTargets {
    entries: Vec<TargetEntry>,
    grand_total: f64,
}

impl ValidatedTargets {
    /// Returns the grand total (sum of any one variable's targets).
    #[must_use]
    pub const fn grand_total(&self) -> f64 {
        self.grand_total
    }

    /// Returns the resolved target entries.
    #[must_use]
    pub fn entries(&self) -> &[TargetEntry] {
        &self.entries
    }

    /// Returns the number of marginal constraints (one per targeted variable).
    #[must_use]
    pub const fn n_constraints(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::survey::Variable;

    fn test_survey() -> CodedSurvey {
        let vars = vec![
            Variable {
                name: "age".into(),
                levels: 3,
                labels: None,
            },
            Variable {
                name: "gender".into(),
                levels: 2,
                labels: None,
            },
        ];
        CodedSurvey::from_flat_codes(vars, vec![0, 0, 1, 1, 2, 0], 3).unwrap()
    }

    #[test]
    fn valid_targets() {
        let survey = test_survey();
        let targets = PopulationTargets::new()
            .add("age", vec![100.0, 200.0, 300.0])
            .add("gender", vec![250.0, 350.0])
            .validate(&survey)
            .unwrap();

        assert_eq!(targets.grand_total(), 600.0);
        assert_eq!(targets.n_constraints(), 2);
        assert_eq!(targets.entries().len(), 2);
    }

    #[test]
    fn single_variable() {
        let survey = test_survey();
        let targets = PopulationTargets::new()
            .add("age", vec![100.0, 200.0, 300.0])
            .validate(&survey)
            .unwrap();

        assert_eq!(targets.grand_total(), 600.0);
        assert_eq!(targets.n_constraints(), 1);
    }

    #[test]
    fn variable_not_found() {
        let survey = test_survey();
        let result = PopulationTargets::new()
            .add("nonexistent", vec![1.0, 2.0])
            .validate(&survey);
        assert!(matches!(
            result,
            Err(RakingError::VariableNotFound { ref name }) if name == "nonexistent"
        ));
    }

    #[test]
    fn target_length_mismatch() {
        let survey = test_survey();
        let result = PopulationTargets::new()
            .add("age", vec![1.0, 2.0]) // age has 3 levels, not 2
            .validate(&survey);
        assert!(matches!(
            result,
            Err(RakingError::TargetLengthMismatch {
                ref name,
                expected: 3,
                got: 2,
            }) if name == "age"
        ));
    }

    #[test]
    fn duplicate_variable() {
        let survey = test_survey();
        let result = PopulationTargets::new()
            .add("age", vec![1.0, 2.0, 3.0])
            .add("age", vec![2.0, 2.0, 2.0])
            .validate(&survey);
        assert!(matches!(
            result,
            Err(RakingError::DuplicateVariable { ref name }) if name == "age"
        ));
    }

    #[test]
    fn inconsistent_totals() {
        let survey = test_survey();
        let result = PopulationTargets::new()
            .add("age", vec![100.0, 200.0, 300.0]) // sum = 600
            .add("gender", vec![100.0, 200.0]) // sum = 300
            .validate(&survey);
        assert!(matches!(
            result,
            Err(RakingError::InconsistentTotals { ref variable_a, ref variable_b, .. })
                if variable_a == "age" && variable_b == "gender"
        ));
    }

    #[test]
    fn no_targets_rejected() {
        let survey = test_survey();
        let result = PopulationTargets::new().validate(&survey);
        assert!(matches!(result, Err(RakingError::NoTargets)));
    }

    #[test]
    fn consistent_within_tolerance() {
        let survey = test_survey();
        let result = PopulationTargets::new()
            .add("age", vec![100.0, 200.0, 300.0]) // sum = 600
            .add("gender", vec![300.0, 300.0 + 1e-9]) // sum ≈ 600
            .validate(&survey);
        assert!(result.is_ok());
    }

    #[test]
    fn population_scale_rounding_tolerated() {
        // Census-style margins rounded to whole people: totals differ by 1
        // out of 300 million. Relative tolerance (1e-6 * G = 300) accepts it.
        let survey = test_survey();
        let result = PopulationTargets::new()
            .add("age", vec![100_000_000.0, 100_000_000.0, 100_000_001.0])
            .add("gender", vec![150_000_000.0, 150_000_000.0])
            .validate(&survey);
        assert!(result.is_ok(), "rounded population totals should validate");
    }

    #[test]
    fn custom_tolerance_is_relative() {
        // Same 1-person mismatch, but a tolerance of 1e-12 scales to a
        // threshold of 3e-4 and rejects it.
        let survey = test_survey();
        let result = PopulationTargets::new()
            .add("age", vec![100_000_000.0, 100_000_000.0, 100_000_001.0])
            .add("gender", vec![150_000_000.0, 150_000_000.0])
            .validate_with_tolerance(&survey, 1e-12);
        assert!(matches!(
            result,
            Err(RakingError::InconsistentTotals { .. })
        ));
    }

    #[test]
    fn small_totals_use_absolute_scale() {
        // Grand total < 1: threshold clamps to tolerance * 1, so a 0.5
        // mismatch against a 0.6 total is still caught.
        let survey = test_survey();
        let result = PopulationTargets::new()
            .add("age", vec![0.2, 0.2, 0.2]) // sum = 0.6
            .add("gender", vec![0.05, 0.05]) // sum = 0.1
            .validate(&survey);
        assert!(matches!(
            result,
            Err(RakingError::InconsistentTotals { .. })
        ));
    }

    #[test]
    fn negative_target_rejected() {
        let survey = test_survey();
        let result = PopulationTargets::new()
            .add("age", vec![1.0, -2.0, 3.0])
            .validate(&survey);
        assert!(matches!(
            result,
            Err(RakingError::InvalidTarget { ref name, index: 1, .. }) if name == "age"
        ));
    }

    #[test]
    fn nan_target_rejected() {
        let survey = test_survey();
        let result = PopulationTargets::new()
            .add("age", vec![1.0, f64::NAN, 3.0])
            .validate(&survey);
        assert!(matches!(
            result,
            Err(RakingError::InvalidTarget { index: 1, .. })
        ));
    }

    #[test]
    fn grand_total_from_first_entry() {
        let survey = test_survey();
        let targets = PopulationTargets::new()
            .add("gender", vec![5.0, 10.0])
            .add("age", vec![3.0, 4.0, 8.0])
            .validate(&survey)
            .unwrap();
        assert_eq!(targets.grand_total(), 15.0);
    }
}
