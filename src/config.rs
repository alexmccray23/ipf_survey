use ipf::ConvergenceConfig;

use crate::RakingError;

/// How to normalize final weights.
#[derive(Debug, Clone, Copy)]
pub enum Normalization {
    /// Raw weights from IPF (sum = grand total from targets).
    None,
    /// Scale so weights sum to number of records.
    SumToN,
    /// Scale so weights sum to the given value.
    SumTo(f64),
    /// Scale so mean weight = 1.0 (equivalent to `SumToN`).
    MeanOne,
}

/// Controls raking algorithm behavior.
#[derive(Debug, Clone)]
pub struct RakingConfig {
    /// Convergence settings for the IPF solver.
    pub convergence: ConvergenceConfig<f64>,
    /// Optional (lower, upper) bounds for the adjustment factor.
    pub weight_bounds: Option<(f64, f64)>,
    /// Maximum trim-rerake cycles before giving up.
    pub max_trim_cycles: usize,
    /// Stop trimming when max adjustment change < this.
    pub trim_tolerance: f64,
    /// Weight normalization mode.
    pub normalization: Normalization,
    /// Whether to record per-iteration diagnostics in the IPF solver.
    pub diagnostics: bool,
}

impl Default for RakingConfig {
    fn default() -> Self {
        Self {
            convergence: ConvergenceConfig::default(),
            weight_bounds: None,
            max_trim_cycles: 50,
            trim_tolerance: 1e-6,
            normalization: Normalization::None,
            diagnostics: false,
        }
    }
}

impl RakingConfig {
    /// Validate the configuration, returning an error if bounds are invalid.
    pub fn validate(&self) -> Result<(), RakingError> {
        if let Some((lower, upper)) = self.weight_bounds
            && (lower < 0.0 || lower > upper)
        {
            return Err(RakingError::InvalidBounds { lower, upper });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = RakingConfig::default();
        assert!(cfg.weight_bounds.is_none());
        assert_eq!(cfg.max_trim_cycles, 50);
        assert!(!cfg.diagnostics);
        assert!(matches!(cfg.normalization, Normalization::None));
    }

    #[test]
    fn valid_bounds() {
        let cfg = RakingConfig {
            weight_bounds: Some((0.5, 3.0)),
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn valid_bounds_equal() {
        let cfg = RakingConfig {
            weight_bounds: Some((1.0, 1.0)),
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn invalid_bounds_reversed() {
        let cfg = RakingConfig {
            weight_bounds: Some((5.0, 1.0)),
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(RakingError::InvalidBounds { .. })
        ));
    }

    #[test]
    fn invalid_bounds_negative_lower() {
        let cfg = RakingConfig {
            weight_bounds: Some((-1.0, 3.0)),
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(RakingError::InvalidBounds { .. })
        ));
    }

    #[test]
    fn no_bounds_is_valid() {
        let cfg = RakingConfig::default();
        assert!(cfg.validate().is_ok());
    }
}
