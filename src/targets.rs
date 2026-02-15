use crate::survey::CodedSurvey;
use crate::RakingError;

/// A single marginal-target entry: which variable and what target values.
#[derive(Debug, Clone)]
pub struct TargetEntry {
    pub variable_index: usize,
    pub targets: Vec<f64>,
}

/// Builder for assembling population targets before validation.
#[derive(Debug, Clone)]
pub struct PopulationTargets {
    entries: Vec<TargetEntry>,
}

impl PopulationTargets {
    #[must_use] 
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add marginal targets for a variable (infallible builder method).
    #[must_use] 
    pub fn add(mut self, variable_index: usize, targets: Vec<f64>) -> Self {
        self.entries.push(TargetEntry {
            variable_index,
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
        // Check for duplicate variable indices
        for (i, a) in self.entries.iter().enumerate() {
            for b in &self.entries[i + 1..] {
                if a.variable_index == b.variable_index {
                    return Err(RakingError::DuplicateVariable {
                        variable: a.variable_index,
                    });
                }
            }
        }

        // Validate each entry
        for entry in &self.entries {
            // Variable index in range
            if entry.variable_index >= survey.n_variables() {
                return Err(RakingError::VariableIndexOutOfRange {
                    index: entry.variable_index,
                    n_variables: survey.n_variables(),
                });
            }

            // Target length matches levels
            let expected = survey.variables()[entry.variable_index].levels;
            if entry.targets.len() != expected {
                return Err(RakingError::TargetLengthMismatch {
                    variable: entry.variable_index,
                    expected,
                    got: entry.targets.len(),
                });
            }
        }

        // Compute grand totals and check consistency
        let totals: Vec<f64> = self
            .entries
            .iter()
            .map(|e| e.targets.iter().sum())
            .collect();

        if totals.len() >= 2 {
            let first = totals[0];
            for (i, &total) in totals.iter().enumerate().skip(1) {
                let diff = (first - total).abs();
                if diff > tolerance {
                    return Err(RakingError::InconsistentTotals {
                        variable_a: self.entries[0].variable_index,
                        variable_b: self.entries[i].variable_index,
                        diff,
                    });
                }
            }
        }

        let grand_total = totals.first().copied().unwrap_or(0.0);

        Ok(ValidatedTargets {
            entries: self.entries,
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
            .add(0, vec![100.0, 200.0, 300.0])
            .add(1, vec![250.0, 350.0])
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
            .add(0, vec![100.0, 200.0, 300.0])
            .validate(&survey)
            .unwrap();

        assert_eq!(targets.grand_total(), 600.0);
        assert_eq!(targets.n_constraints(), 1);
    }

    #[test]
    fn variable_out_of_range() {
        let survey = test_survey();
        let result = PopulationTargets::new()
            .add(5, vec![1.0, 2.0])
            .validate(&survey);
        assert!(matches!(
            result,
            Err(RakingError::VariableIndexOutOfRange {
                index: 5,
                n_variables: 2,
            })
        ));
    }

    #[test]
    fn target_length_mismatch() {
        let survey = test_survey();
        let result = PopulationTargets::new()
            .add(0, vec![1.0, 2.0]) // age has 3 levels, not 2
            .validate(&survey);
        assert!(matches!(
            result,
            Err(RakingError::TargetLengthMismatch {
                variable: 0,
                expected: 3,
                got: 2,
            })
        ));
    }

    #[test]
    fn duplicate_variable() {
        let survey = test_survey();
        let result = PopulationTargets::new()
            .add(0, vec![1.0, 2.0, 3.0])
            .add(0, vec![2.0, 2.0, 2.0])
            .validate(&survey);
        assert!(matches!(
            result,
            Err(RakingError::DuplicateVariable { variable: 0 })
        ));
    }

    #[test]
    fn inconsistent_totals() {
        let survey = test_survey();
        let result = PopulationTargets::new()
            .add(0, vec![100.0, 200.0, 300.0]) // sum = 600
            .add(1, vec![100.0, 200.0]) // sum = 300
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
            .add(0, vec![100.0, 200.0, 300.0]) // sum = 600
            .add(1, vec![300.0, 300.0 + 1e-9]) // sum ≈ 600
            .validate(&survey);
        assert!(result.is_ok());
    }

    #[test]
    fn grand_total_from_first_entry() {
        let survey = test_survey();
        let targets = PopulationTargets::new()
            .add(1, vec![5.0, 10.0])
            .add(0, vec![3.0, 4.0, 8.0])
            .validate(&survey)
            .unwrap();
        assert_eq!(targets.grand_total(), 15.0);
    }
}
