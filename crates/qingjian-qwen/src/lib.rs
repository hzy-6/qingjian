//! Qwen 小模型的本地推理（llama.cpp）：给整句路径按 BPE token 打 log 概率。
//!
//! 加载 GGUF（如 `Qwen3.5-2B-UD-Q4_K_XL-text.gguf`），给「光标前文 + 候选文本」按 token 累加 log 概率，
//! 实现 Core 的 [`SentenceScorer`]，给整句前几条路径做重打分。
//!
//! 需 `runtime` feature（编译 llama.cpp，Apple GPU 加速；`QINGJIAN_QWEN_GPU_LAYERS=0` 可强制纯 CPU）；
//! 不开 feature 时是空壳，workspace 默认构建不拉 llama.cpp。
//! 每条候选拼上前文各自成一个序列、一次 decode 打一批，打完清空 KV——Qwen3.5 是注意力 + SSM 的
//! 混合架构，`seq_cp` 与中间位置回卷都不可用，整序列重算最简单可靠（前文几十个 token 在 Metal 上很便宜）。

#![cfg(feature = "runtime")]

mod error;
mod scorer;

pub use error::QwenError;
pub use scorer::QwenScorer;
