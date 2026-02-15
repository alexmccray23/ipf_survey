#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::cast_precision_loss
)]

mod config;
mod diagnostics;
mod error;
mod rake;
mod survey;
mod tabulate;
mod targets;
mod weights;

pub use config::{Normalization, RakingConfig};
pub use diagnostics::{RakingDiagnostics, WeightSummary};
pub use error::RakingError;
pub use rake::{RakingResult, rake, rake_simple};
pub use survey::{CodedSurvey, SurveyBuilder, Variable};
pub use targets::{PopulationTargets, TargetEntry, ValidatedTargets};
