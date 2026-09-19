//! Qwen 小模型（GGUF，llama.cpp 推理）的错误类型。

use std::path::PathBuf;

/// Qwen 打分器加载与推理失败的统一错误。
#[derive(Debug, thiserror::Error)]
pub enum QwenError {
    /// GGUF 文件不存在。
    #[error("model file not found: {0}")]
    NotFound(PathBuf),

    /// llama.cpp 后端初始化失败（一个进程只允许一次）。
    #[error("llama.cpp backend init failed: {0}")]
    Backend(String),

    /// 模型加载失败：文件损坏、架构不支持或显存不足。
    #[error("model load failed: {0}")]
    Load(String),

    /// 推理上下文创建失败。
    #[error("context creation failed: {0}")]
    Context(String),

    /// 文本切不出 token。
    #[error("tokenize failed: {0}")]
    Tokenize(String),

    /// 一次解码失败。
    #[error("decode failed: {0}")]
    Decode(String),

    /// 前文加最长候选超过上下文窗口。
    #[error("sequence too long: prefix {prefix} + longest {longest} tokens")]
    TooLong { prefix: usize, longest: usize },
}
