/// 整句转换的打分来源：词级语言模型。实现放兄弟 crate（`qingjian-lm`），Core 只认这个 trait。
pub trait LanguageModel: Send {
    /// `log P(word | previous)`；`previous` 为 `None` 表示句首。模型不认识 `word` 时返回 `None`，
    /// 由 Core 用词库词频兜底。
    fn log_prob(&self, previous: Option<&str>, word: &str) -> Option<f64>;

    /// 枚举 `previous` 的后继词（按计数从高到低，至多 `limit` 条），给本地联想出接续提议用。
    /// 实现可以不支持（返回空），联想退化成只靠个人 n-gram。
    fn successors(&self, previous: &str, limit: usize) -> Vec<(String, u32)> {
        let _ = (previous, limit);
        Vec::new()
    }
}

/// 没接语言模型：一律兜底，整句转换退化为一元词频。
#[derive(Debug, Default, Clone, Copy)]
pub struct NoLanguageModel;

impl LanguageModel for NoLanguageModel {
    fn log_prob(&self, _previous: Option<&str>, _word: &str) -> Option<f64> {
        None
    }
}
