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

    /// Validate against a survey with default tolerance (1e-6).
    pub fn validate(self, survey: &CodedSurvey) -> Result<ValidatedTargets, RakingError> {
        self.validate_with_tolerance(survey, 1e-6)
    }

    /// Validate against a survey with a custom grand-total tolerance.
    pub fn validate_with_tolerance(
        self,
        survey: &CodedSurvey,
        tolerance: f64,
    ) -> Result<ValidatedTargets, RakingError> {
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
        for entry in &self.entries {
            let var_index = survey
                .variables()
                .iter()
                .position(|v| v.name == entry.variable_name)
                .ok_or_else(|| RakingError::VariableNotFound {
                    name: entry.variable_name.clone(),
                })?;

            let expected = survey.variables()[var_index].levels;
            if entry.targets.len() != expected {
                return Err(RakingError::TargetLengthMismatch {
                    name: entry.variable_name.clone(),
                    expected,
                    got: entry.targets.len(),
                });
            }

            resolved.push(TargetEntry {
                variable_index: var_index,
                targets: entry.targets.clone(),
            });
        }

        // Compute grand totals and check consistency
        let totals: Vec<f64> = resolved.iter().map(|e| e.targets.iter().sum()).collect();

        if totals.len() >= 2 {
            let first = totals[0];
            for (i, &total) in totals.iter().enumerate().skip(1) {
                let diff = (first - total).abs();
                if diff > tolerance {
                    return Err(RakingError::InconsistentTotals {
                        variable_a: resolved[0].variable_index,
                        variable_b: resolved[i].variable_index,
                        diff,
                    });
                }
            }
        }

        let grand_total = totals.first().copied().unwrap_or(0.0);

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
    #[must_use]
    pub const fn grand_total(&self) -> f64 {
        self.grand_total
    }

    #[must_use]
    pub fn entries(&self) -> &[TargetEntry] {
        &self.entries
    }

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
            Err(RakingError::InconsistentTotals { .. })
        ));
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
