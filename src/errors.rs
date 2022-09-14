//! Definition of errors.

use core::fmt;

#[cfg(feature = "std")]
use std::error::Error;

/// Error used when the argument is invalid.
#[derive(Debug)]
pub struct InvalidArgumentError {
    msg: &'static str,
}

/// The error type for Rucrf.
#[derive(Debug)]
pub enum RucrfError {
    InvalidArgument(InvalidArgumentError),
}

impl fmt::Display for InvalidArgumentError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "InvalidArgumentError: {}", self.msg)
    }
}

#[cfg(feature = "std")]
impl Error for InvalidArgumentError {}

impl RucrfError {
    /// Creates a new [`InvalidArgumentError`].
    pub const fn invalid_argument(msg: &'static str) -> Self {
        Self::InvalidArgument(InvalidArgumentError { msg })
    }
}

impl fmt::Display for RucrfError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidArgument(e) => e.fmt(f),
        }
    }
}

#[cfg(feature = "std")]
impl Error for RucrfError {}

/// A specialized Result type.
pub type Result<T, E = RucrfError> = core::result::Result<T, E>;
