use crate::RakingError;

/// Describes one categorical variable used for raking.
#[derive(Debug, Clone)]
pub struct Variable {
    pub name: String,
    pub levels: usize,
    pub labels: Option<Vec<String>>,
}

/// Flat, cache-friendly representation of N records x K variables.
///
/// Codes are stored row-major: `codes[record * n_variables + var] = category index`.
#[derive(Debug)]
pub struct CodedSurvey {
    n_records: usize,
    variables: Vec<Variable>,
    codes: Vec<usize>,
    base_weights: Option<Vec<f64>>,
}

impl CodedSurvey {
    /// Create a builder for constructing a survey record-by-record.
    #[must_use]
    pub const fn builder(variables: Vec<Variable>) -> SurveyBuilder {
        SurveyBuilder {
            variables,
            codes: Vec::new(),
            base_weights: None,
            n_records: 0,
        }
    }

    /// Bulk-load from a flat row-major codes array.
    pub fn from_flat_codes(
        variables: Vec<Variable>,
        codes: Vec<usize>,
        n_records: usize,
    ) -> Result<Self, RakingError> {
        Self::from_flat_codes_inner(variables, codes, n_records, None)
    }

    /// Bulk-load from a flat row-major codes array with base weights.
    pub fn from_flat_codes_weighted(
        variables: Vec<Variable>,
        codes: Vec<usize>,
        n_records: usize,
        base_weights: Vec<f64>,
    ) -> Result<Self, RakingError> {
        Self::from_flat_codes_inner(variables, codes, n_records, Some(base_weights))
    }

    fn from_flat_codes_inner(
        variables: Vec<Variable>,
        codes: Vec<usize>,
        n_records: usize,
        base_weights: Option<Vec<f64>>,
    ) -> Result<Self, RakingError> {
        if n_records == 0 {
            return Err(RakingError::EmptySurvey);
        }

        let k = variables.len();

        if codes.len() != n_records * k {
            return Err(RakingError::BaseWeightLength {
                expected: n_records * k,
                got: codes.len(),
            });
        }

        // Validate labels
        for (v, var) in variables.iter().enumerate() {
            if let Some(ref labels) = var.labels
                && labels.len() != var.levels
            {
                return Err(RakingError::LabelMismatch {
                    variable: v,
                    levels: var.levels,
                    labels: labels.len(),
                });
            }
        }

        // Validate all codes
        for r in 0..n_records {
            for v in 0..k {
                let code = codes[r * k + v];
                if code >= variables[v].levels {
                    return Err(RakingError::InvalidCode {
                        record: r,
                        variable: v,
                        code,
                        max: variables[v].levels - 1,
                    });
                }
            }
        }

        // Validate base weights
        if let Some(ref bw) = base_weights {
            if bw.len() != n_records {
                return Err(RakingError::BaseWeightLength {
                    expected: n_records,
                    got: bw.len(),
                });
            }
            for (r, &w) in bw.iter().enumerate() {
                if w < 0.0 {
                    return Err(RakingError::NegativeBaseWeight {
                        record: r,
                        weight: w,
                    });
                }
            }
        }

        Ok(Self {
            n_records,
            variables,
            codes,
            base_weights,
        })
    }

    #[must_use]
    pub const fn n_records(&self) -> usize {
        self.n_records
    }

    #[must_use]
    pub const fn n_variables(&self) -> usize {
        self.variables.len()
    }

    #[must_use]
    pub fn variables(&self) -> &[Variable] {
        &self.variables
    }

    #[must_use]
    pub fn record_codes(&self, record_index: usize) -> &[usize] {
        let k = self.variables.len();
        &self.codes[record_index * k..(record_index + 1) * k]
    }

    #[must_use]
    pub fn base_weight(&self, record_index: usize) -> f64 {
        self.base_weights
            .as_ref()
            .map_or(1.0, |bw| bw[record_index])
    }

    #[must_use]
    pub fn flat_codes(&self) -> &[usize] {
        &self.codes
    }

    #[must_use]
    pub fn base_weights(&self) -> Option<&[f64]> {
        self.base_weights.as_deref()
    }
}

/// Builder for constructing a [`CodedSurvey`] record-by-record.
#[derive(Debug)]
pub struct SurveyBuilder {
    variables: Vec<Variable>,
    codes: Vec<usize>,
    base_weights: Option<Vec<f64>>,
    n_records: usize,
}

impl SurveyBuilder {
    /// Append one unweighted record.
    pub fn push_record(&mut self, codes: &[usize]) -> Result<&mut Self, RakingError> {
        let k = self.variables.len();
        if codes.len() != k {
            return Err(RakingError::BaseWeightLength {
                expected: k,
                got: codes.len(),
            });
        }

        for (v, &code) in codes.iter().enumerate() {
            if code >= self.variables[v].levels {
                return Err(RakingError::InvalidCode {
                    record: self.n_records,
                    variable: v,
                    code,
                    max: self.variables[v].levels - 1,
                });
            }
        }

        self.codes.extend_from_slice(codes);

        // If we already started tracking weights, fill 1.0 for this record
        if let Some(ref mut bw) = self.base_weights {
            bw.push(1.0);
        }

        self.n_records += 1;
        Ok(self)
    }

    /// Append one record with a base weight.
    pub fn push_record_weighted(
        &mut self,
        codes: &[usize],
        base_weight: f64,
    ) -> Result<&mut Self, RakingError> {
        if base_weight < 0.0 {
            return Err(RakingError::NegativeBaseWeight {
                record: self.n_records,
                weight: base_weight,
            });
        }

        let k = self.variables.len();
        if codes.len() != k {
            return Err(RakingError::BaseWeightLength {
                expected: k,
                got: codes.len(),
            });
        }

        for (v, &code) in codes.iter().enumerate() {
            if code >= self.variables[v].levels {
                return Err(RakingError::InvalidCode {
                    record: self.n_records,
                    variable: v,
                    code,
                    max: self.variables[v].levels - 1,
                });
            }
        }

        self.codes.extend_from_slice(codes);

        // Retroactively fill 1.0s for prior records if this is the first weighted push
        if self.base_weights.is_none() {
            self.base_weights = Some(vec![1.0; self.n_records]);
        }
        self.base_weights.as_mut().unwrap().push(base_weight);

        self.n_records += 1;
        Ok(self)
    }

    /// Finalize the builder, returning a [`CodedSurvey`].
    pub fn build(self) -> Result<CodedSurvey, RakingError> {
        if self.n_records == 0 {
            return Err(RakingError::EmptySurvey);
        }

        // Validate labels
        for (v, var) in self.variables.iter().enumerate() {
            if let Some(ref labels) = var.labels
                && labels.len() != var.levels
            {
                return Err(RakingError::LabelMismatch {
                    variable: v,
                    levels: var.levels,
                    labels: labels.len(),
                });
            }
        }

        Ok(CodedSurvey {
            n_records: self.n_records,
            variables: self.variables,
            codes: self.codes,
            base_weights: self.base_weights,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_variables() -> Vec<Variable> {
        vec![
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
        ]
    }

    #[test]
    fn builder_basic() {
        let mut builder = CodedSurvey::builder(test_variables());
        builder.push_record(&[0, 0]).unwrap();
        builder.push_record(&[1, 1]).unwrap();
        builder.push_record(&[2, 0]).unwrap();
        let survey = builder.build().unwrap();

        assert_eq!(survey.n_records(), 3);
        assert_eq!(survey.n_variables(), 2);
        assert_eq!(survey.record_codes(0), &[0, 0]);
        assert_eq!(survey.record_codes(1), &[1, 1]);
        assert_eq!(survey.record_codes(2), &[2, 0]);
        assert_eq!(survey.base_weight(0), 1.0);
        assert!(survey.base_weights().is_none());
    }

    #[test]
    fn builder_weighted() {
        let mut builder = CodedSurvey::builder(test_variables());
        builder.push_record_weighted(&[0, 0], 2.0).unwrap();
        builder.push_record_weighted(&[1, 1], 0.5).unwrap();
        let survey = builder.build().unwrap();

        assert_eq!(survey.base_weight(0), 2.0);
        assert_eq!(survey.base_weight(1), 0.5);
        assert!(survey.base_weights().is_some());
    }

    #[test]
    fn builder_mixed_weights() {
        let mut builder = CodedSurvey::builder(test_variables());
        builder.push_record(&[0, 0]).unwrap();
        builder.push_record_weighted(&[1, 1], 2.0).unwrap();
        builder.push_record(&[2, 0]).unwrap();
        let survey = builder.build().unwrap();

        assert_eq!(survey.base_weight(0), 1.0);
        assert_eq!(survey.base_weight(1), 2.0);
        assert_eq!(survey.base_weight(2), 1.0);
    }

    #[test]
    fn invalid_code() {
        let mut builder = CodedSurvey::builder(test_variables());
        let result = builder.push_record(&[3, 0]);
        assert!(matches!(
            result,
            Err(RakingError::InvalidCode {
                record: 0,
                variable: 0,
                code: 3,
                max: 2,
            })
        ));
    }

    #[test]
    fn wrong_code_count() {
        let mut builder = CodedSurvey::builder(test_variables());
        let result = builder.push_record(&[0]);
        assert!(matches!(
            result,
            Err(RakingError::BaseWeightLength {
                expected: 2,
                got: 1,
            })
        ));
    }

    #[test]
    fn negative_weight() {
        let mut builder = CodedSurvey::builder(test_variables());
        let result = builder.push_record_weighted(&[0, 0], -1.0);
        assert!(matches!(
            result,
            Err(RakingError::NegativeBaseWeight { record: 0, .. })
        ));
    }

    #[test]
    fn empty_survey() {
        let builder = CodedSurvey::builder(test_variables());
        let result = builder.build();
        assert!(matches!(result, Err(RakingError::EmptySurvey)));
    }

    #[test]
    fn label_mismatch() {
        let vars = vec![Variable {
            name: "x".into(),
            levels: 3,
            labels: Some(vec!["a".into(), "b".into()]),
        }];
        let mut builder = CodedSurvey::builder(vars);
        builder.push_record(&[0]).unwrap();
        let result = builder.build();
        assert!(matches!(result, Err(RakingError::LabelMismatch { .. })));
    }

    #[test]
    fn from_flat_codes() {
        let survey =
            CodedSurvey::from_flat_codes(test_variables(), vec![0, 0, 1, 1, 2, 0], 3).unwrap();
        assert_eq!(survey.n_records(), 3);
        assert_eq!(survey.record_codes(1), &[1, 1]);
        assert_eq!(survey.flat_codes(), &[0, 0, 1, 1, 2, 0]);
    }

    #[test]
    fn from_flat_codes_weighted() {
        let survey = CodedSurvey::from_flat_codes_weighted(
            test_variables(),
            vec![0, 0, 1, 1],
            2,
            vec![1.5, 2.5],
        )
        .unwrap();
        assert_eq!(survey.base_weight(0), 1.5);
        assert_eq!(survey.base_weight(1), 2.5);
    }

    #[test]
    fn from_flat_codes_wrong_length() {
        let result = CodedSurvey::from_flat_codes(test_variables(), vec![0, 0, 1], 2);
        assert!(matches!(result, Err(RakingError::BaseWeightLength { .. })));
    }

    #[test]
    fn from_flat_codes_empty() {
        let result = CodedSurvey::from_flat_codes(test_variables(), vec![], 0);
        assert!(matches!(result, Err(RakingError::EmptySurvey)));
    }

    #[test]
    fn zero_weight_allowed() {
        let mut builder = CodedSurvey::builder(test_variables());
        builder.push_record_weighted(&[0, 0], 0.0).unwrap();
        let survey = builder.build().unwrap();
        assert_eq!(survey.base_weight(0), 0.0);
    }

    #[test]
    fn accessors() {
        let vars = vec![
            Variable {
                name: "a".into(),
                levels: 2,
                labels: Some(vec!["x".into(), "y".into()]),
            },
            Variable {
                name: "b".into(),
                levels: 3,
                labels: None,
            },
        ];
        let survey = CodedSurvey::from_flat_codes(vars, vec![0, 0, 1, 2, 0, 1], 3).unwrap();
        assert_eq!(survey.variables()[0].name, "a");
        assert_eq!(survey.variables()[1].levels, 3);
        assert_eq!(survey.n_variables(), 2);
    }
}
