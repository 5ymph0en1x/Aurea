//! AUREA codec error types.
//! Replaces bare unwrap() and string errors with typed errors.

use std::fmt;

#[derive(Debug)]
pub enum AureaError {
    /// Invalid file format (bad magic, truncated header)
    InvalidFormat(String),
    /// Unsupported codec version
    UnsupportedVersion(u8),
    /// Decoding failure (corrupt data, unexpected EOF)
    DecodeFailed(String),
    /// Encoding failure
    EncodeFailed(String),
    /// I/O error
    Io(std::io::Error),
    /// Dimension mismatch
    DimensionMismatch { expected: usize, got: usize },
}

impl fmt::Display for AureaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AureaError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            AureaError::UnsupportedVersion(v) => write!(f, "Unsupported version: {}", v),
            AureaError::DecodeFailed(msg) => write!(f, "Decode failed: {}", msg),
            AureaError::EncodeFailed(msg) => write!(f, "Encode failed: {}", msg),
            AureaError::Io(e) => write!(f, "I/O error: {}", e),
            AureaError::DimensionMismatch { expected, got } =>
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got),
        }
    }
}

impl std::error::Error for AureaError {}

impl From<std::io::Error> for AureaError {
    fn from(e: std::io::Error) -> Self {
        AureaError::Io(e)
    }
}

impl From<String> for AureaError {
    fn from(s: String) -> Self {
        AureaError::DecodeFailed(s)
    }
}

impl From<&str> for AureaError {
    fn from(s: &str) -> Self {
        AureaError::DecodeFailed(s.to_string())
    }
}
