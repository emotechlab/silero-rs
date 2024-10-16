//! All possible errors VAD might encounter.
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VadError {
    #[error("Float sample must be in the range -1.0 to 1.0")]
    FloatSampleNotInRangeMinus1To1,
}
