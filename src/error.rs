use std::fmt;

use ipf::IpfError;

/// All errors that can occur in the `ipf_survey` crate.
#[derive(Debug)]
pub enum RakingError {
    // -- Survey construction errors --
    /// A record's code is out of range for its variable.
    InvalidCode {
        record: usize,
        variable: usize,
        code: usize,
        max: usize,
    },

    /// Variable label count doesn't match level count.
    LabelMismatch {
        variable: usize,
        levels: usize,
        labels: usize,
    },

    /// Base weights length doesn't match record count.
    BaseWeightLength { expected: usize, got: usize },

    /// Negative base weight.
    NegativeBaseWeight { record: usize, weight: f64 },

    /// No records provided.
    EmptySurvey,

    // -- Target errors --
    /// Variable index out of range.
    VariableIndexOutOfRange { index: usize, n_variables: usize },

    /// Target vector length doesn't match variable levels.
    TargetLengthMismatch {
        variable: usize,
        expected: usize,
        got: usize,
    },

    /// Same variable targeted twice.
    DuplicateVariable { variable: usize },

    /// Grand totals across variables are inconsistent.
    InconsistentTotals {
        variable_a: usize,
        variable_b: usize,
        diff: f64,
    },

    // -- Algorithm errors --
    /// IPF solver error (wraps `ipf::IpfError`).
    Ipf(IpfError<f64>),

    /// Weight trimming did not converge within `max_trim_cycles`.
    TrimNotConverged { cycles: usize },

    // -- Config errors --
    /// Invalid weight bounds (lower > upper, or negative).
    InvalidBounds { lower: f64, upper: f64 },
}

impl From<IpfError<f64>> for RakingError {
    fn from(err: IpfError<f64>) -> Self {
        Self::Ipf(err)
    }
}

impl fmt::Display for RakingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCode {
                record,
                variable,
                code,
                max,
            } => write!(
                f,
                "record {record}: variable {variable} code {code} out of range (max {max})"
            ),
            Self::LabelMismatch {
                variable,
                levels,
                labels,
            } => write!(
                f,
                "variable {variable}: {labels} labels but {levels} levels"
            ),
            Self::BaseWeightLength { expected, got } => {
                write!(f, "expected {expected} base weights, got {got}")
            }
            Self::NegativeBaseWeight { record, weight } => {
                write!(f, "record {record}: negative base weight {weight}")
            }
            Self::EmptySurvey => write!(f, "survey has no records"),
            Self::VariableIndexOutOfRange { index, n_variables } => write!(
                f,
                "variable index {index} out of range (survey has {n_variables} variables)"
            ),
            Self::TargetLengthMismatch {
                variable,
                expected,
                got,
            } => write!(
                f,
                "variable {variable}: expected {expected} targets, got {got}"
            ),
            Self::DuplicateVariable { variable } => {
                write!(f, "duplicate targets for variable {variable}")
            }
            Self::InconsistentTotals {
                variable_a,
                variable_b,
                diff,
            } => write!(
                f,
                "inconsistent grand totals: variables {variable_a} and {variable_b} differ by {diff}"
            ),
            Self::Ipf(e) => write!(f, "IPF solver error: {e:?}"),
            Self::TrimNotConverged { cycles } => {
                write!(f, "weight trimming did not converge after {cycles} cycles")
            }
            Self::InvalidBounds { lower, upper } => {
                write!(f, "invalid weight bounds: [{lower}, {upper}]")
            }
        }
    }
}

impl std::error::Error for RakingError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_ipf_error() {
        let ipf_err = IpfError::<f64>::NegativeTarget { axis: 0, index: 1 };
        let raking_err: RakingError = ipf_err.into();
        assert!(matches!(raking_err, RakingError::Ipf(_)));
    }

    #[test]
    fn display_all_variants() {
        let variants: Vec<RakingError> = vec![
            RakingError::InvalidCode {
                record: 0,
                variable: 1,
                code: 5,
                max: 3,
            },
            RakingError::LabelMismatch {
                variable: 0,
                levels: 3,
                labels: 2,
            },
            RakingError::BaseWeightLength {
                expected: 10,
                got: 8,
            },
            RakingError::NegativeBaseWeight {
                record: 2,
                weight: -1.0,
            },
            RakingError::EmptySurvey,
            RakingError::VariableIndexOutOfRange {
                index: 5,
                n_variables: 3,
            },
            RakingError::TargetLengthMismatch {
                variable: 0,
                expected: 3,
                got: 2,
            },
            RakingError::DuplicateVariable { variable: 1 },
            RakingError::InconsistentTotals {
                variable_a: 0,
                variable_b: 1,
                diff: 0.5,
            },
            RakingError::Ipf(IpfError::NegativeTarget { axis: 0, index: 0 }),
            RakingError::TrimNotConverged { cycles: 50 },
            RakingError::InvalidBounds {
                lower: 5.0,
                upper: 1.0,
            },
        ];

        for v in &variants {
            let msg = v.to_string();
            assert!(!msg.is_empty(), "empty Display for {v:?}");
        }
    }

    #[test]
    fn error_trait_impl() {
        let err: Box<dyn std::error::Error> = Box::new(RakingError::EmptySurvey);
        assert!(!err.to_string().is_empty());
    }
}
