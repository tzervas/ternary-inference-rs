//! Error types for ternary inference.

/// Result alias using the crate's error type.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors from ternary inference operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Model configuration error.
    #[error("config error: {0}")]
    Config(String),

    /// Weight loading error.
    #[error("loading error: {0}")]
    Loading(String),

    /// Shape mismatch during unpacking or inference.
    #[error("shape mismatch: expected {expected}, got {actual}")]
    Shape {
        expected: String,
        actual: String,
    },

    /// Tensor operation error.
    #[error("tensor error: {0}")]
    Tensor(#[from] candle_core::Error),

    /// IO error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parsing error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// SafeTensors error.
    #[error("safetensors error: {0}")]
    SafeTensors(String),

    /// Unsupported architecture.
    #[error("unsupported architecture: {0}")]
    UnsupportedArch(String),
}
