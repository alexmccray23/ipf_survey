use std::fmt;

use ipf::IpfError;

/// All errors that can occur in the `ipf_survey` crate.
#[derive(Debug)]
#[non_exhaustive]
pub enum RakingError {
    // -- Survey construction errors --
    /// A record's code is out of range for its variable.
    InvalidCode {
        record: usize,
        variable: usize,
        code: usize,
        max: usize,
    },

    /// A record's code count doesn't match the number of variables.
    CodeLengthMismatch { expected: usize, got: usize },

    /// Survey has no variables.
    NoVariables,

    /// A variable has zero levels, so no record can carry a valid code for it.
    ZeroLevels { variable: usize },

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

    /// Base weight is NaN or infinite.
    NonFiniteBaseWeight { record: usize },

    /// No records provided.
    EmptySurvey,

    // -- Target errors --
    /// Variable name not found in survey.
    VariableNotFound { name: String },

    /// Target vector length doesn't match variable levels.
    TargetLengthMismatch {
        name: String,
        expected: usize,
        got: usize,
    },

    /// Same variable targeted twice.
    DuplicateVariable { name: String },

    /// A target value is negative, NaN, or infinite.
    InvalidTarget {
        name: String,
        index: usize,
        value: f64,
    },

    /// Grand totals across variables are inconsistent.
    InconsistentTotals {
        variable_a: usize,
        variable_b: usize,
        diff: f64,
    },

    // -- Algorithm errors --
    /// IPF solver error (wraps `ipf::IpfError`).
    Ipf(IpfError),

    /// Weight trimming did not converge within `max_trim_cycles`.
    TrimNotConverged { cycles: usize },

    /// Weight trimming cannot satisfy the targets: the mass frozen at the
    /// bounds conflicts with the marginal target for this variable level.
    TrimInfeasible { variable: usize, level: usize },

    // -- Config errors --
    /// Invalid weight bounds (lower > upper, negative, or NaN).
    InvalidBounds { lower: f64, upper: f64 },

    /// Invalid normalization total (must be finite and positive).
    InvalidNormalization { value: f64 },
}

impl From<IpfError> for RakingError {
    fn from(err: IpfError) -> Self {
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
            Self::CodeLengthMismatch { expected, got } => {
                write!(f, "record has {got} codes, expected {expected}")
            }
            Self::NoVariables => write!(f, "survey has no variables"),
            Self::ZeroLevels { variable } => {
                write!(f, "variable {variable} has zero levels")
            }
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
            Self::NonFiniteBaseWeight { record } => {
                write!(f, "record {record}: base weight is NaN or infinite")
            }
            Self::EmptySurvey => write!(f, "survey has no records"),
            Self::VariableNotFound { name } => {
                write!(f, "variable \"{name}\" not found in survey")
            }
            Self::TargetLengthMismatch {
                name,
                expected,
                got,
            } => write!(
                f,
                "variable \"{name}\": expected {expected} targets, got {got}"
            ),
            Self::DuplicateVariable { name } => {
                write!(f, "duplicate targets for variable \"{name}\"")
            }
            Self::InvalidTarget { name, index, value } => write!(
                f,
                "variable \"{name}\": target[{index}] = {value} is not a finite non-negative number"
            ),
            Self::InconsistentTotals {
                variable_a,
                variable_b,
                diff,
            } => write!(
                f,
                "inconsistent grand totals: variables {variable_a} and {variable_b} differ by {diff}"
            ),
            Self::Ipf(e) => write!(f, "IPF solver error: {e}"),
            Self::TrimNotConverged { cycles } => {
                write!(f, "weight trimming did not converge after {cycles} cycles")
            }
            Self::TrimInfeasible { variable, level } => write!(
                f,
                "weight trimming is infeasible: bounds conflict with the target for variable {variable} level {level}; widen weight_bounds or adjust targets"
            ),
            Self::InvalidBounds { lower, upper } => {
                write!(f, "invalid weight bounds: [{lower}, {upper}]")
            }
            Self::InvalidNormalization { value } => {
                write!(f, "invalid normalization total {value} (must be finite and positive)")
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
        let ipf_err = IpfError::NegativeTarget { axis: 0, index: 1 };
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
            RakingError::VariableNotFound {
                name: "foo".into(),
            },
            RakingError::TargetLengthMismatch {
                name: "age".into(),
                expected: 3,
                got: 2,
            },
            RakingError::DuplicateVariable {
                name: "gender".into(),
            },
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
            RakingError::CodeLengthMismatch {
                expected: 2,
                got: 1,
            },
            RakingError::NoVariables,
            RakingError::ZeroLevels { variable: 0 },
            RakingError::NonFiniteBaseWeight { record: 3 },
            RakingError::InvalidTarget {
                name: "age".into(),
                index: 1,
                value: f64::NAN,
            },
            RakingError::TrimInfeasible {
                variable: 0,
                level: 2,
            },
            RakingError::InvalidNormalization { value: -1.0 },
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
