use std::{error::Error, fmt};

pub const DEFAULT_TEMPERATURE: f32 = 1.0;
pub const DEFAULT_TOP_P: f32 = 1.0;
pub const GREEDY_TEMPERATURE: f32 = 0.0;

pub const INVALID_TEMPERATURE_MESSAGE: &str = "temperature must be finite and in [0, 2]";
pub const INVALID_TOP_P_MESSAGE: &str = "top_p must be finite and in (0, 1]";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingValidationError {
    InvalidTemperature,
    InvalidTopP,
}

impl SamplingValidationError {
    pub const fn message(self) -> &'static str {
        match self {
            Self::InvalidTemperature => INVALID_TEMPERATURE_MESSAGE,
            Self::InvalidTopP => INVALID_TOP_P_MESSAGE,
        }
    }
}

impl fmt::Display for SamplingValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message())
    }
}

impl Error for SamplingValidationError {}

pub fn validate_sampling_controls(
    temperature: Option<f32>,
    top_p: Option<f32>,
) -> Result<(), SamplingValidationError> {
    if let Some(temperature) = temperature
        && (!temperature.is_finite() || !(0.0..=2.0).contains(&temperature))
    {
        return Err(SamplingValidationError::InvalidTemperature);
    }
    if let Some(top_p) = top_p
        && (!top_p.is_finite() || top_p <= 0.0 || top_p > 1.0)
    {
        return Err(SamplingValidationError::InvalidTopP);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_openai_sampling_bounds() {
        validate_sampling_controls(None, None).expect("defaults are valid");
        validate_sampling_controls(Some(0.0), Some(1.0)).expect("greedy is valid");
        validate_sampling_controls(Some(2.0), Some(0.1)).expect("upper temperature is valid");
    }

    #[test]
    fn rejects_invalid_temperature_controls() {
        assert_eq!(
            validate_sampling_controls(Some(-0.1), None),
            Err(SamplingValidationError::InvalidTemperature)
        );
        assert_eq!(
            validate_sampling_controls(Some(f32::NAN), None),
            Err(SamplingValidationError::InvalidTemperature)
        );
        assert_eq!(
            validate_sampling_controls(Some(f32::INFINITY), None),
            Err(SamplingValidationError::InvalidTemperature)
        );
        assert_eq!(
            validate_sampling_controls(Some(2.1), None),
            Err(SamplingValidationError::InvalidTemperature)
        );
    }

    #[test]
    fn rejects_invalid_top_p_controls() {
        assert_eq!(
            validate_sampling_controls(None, Some(0.0)),
            Err(SamplingValidationError::InvalidTopP)
        );
        assert_eq!(
            validate_sampling_controls(None, Some(-0.1)),
            Err(SamplingValidationError::InvalidTopP)
        );
        assert_eq!(
            validate_sampling_controls(None, Some(f32::NAN)),
            Err(SamplingValidationError::InvalidTopP)
        );
        assert_eq!(
            validate_sampling_controls(None, Some(f32::INFINITY)),
            Err(SamplingValidationError::InvalidTopP)
        );
        assert_eq!(
            validate_sampling_controls(None, Some(1.1)),
            Err(SamplingValidationError::InvalidTopP)
        );
    }
}
