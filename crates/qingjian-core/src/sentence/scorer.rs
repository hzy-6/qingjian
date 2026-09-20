/// 整句路径的第二打分来源：给「前文 + 整句文本」按 token 打 log 概率的本地模型（Qwen GGUF，实现在 `qingjian-qwen`）。
/// Core 只认这个 trait；Viterbi 出的前几条路径用它重打分，与路径本身的得分对数线性插值。
pub trait SentenceScorer: Send {
    /// 每条 `texts` 接在 `context`（光标前文，可空）后面的 `log P(text | context)`，与 `texts` 一一对应。
    /// 算不了（模型出错）返回空 Vec，调用方就当没有这个打分。
    fn score(&self, context: &str, texts: &[&str]) -> Vec<f64>;

    /// 可选的双向上下文入口。缺省忽略光标后的文字；支持双向上下文的模型覆盖此方法。
    fn score_with_after(&self, before: &str, _after: &str, texts: &[&str]) -> Vec<f64> {
        self.score(before, texts)
    }

    /// 模型自己建议的单条路径最大神经修正（nat）:打分器声明「该被信多少」（当前实现是推理侧按盲评扫出的经验值）。
    /// 模型没给建议时是 `None`，用引擎的缺省上限 [`crate::NEURAL_MAX_ADJUSTMENT`]。
    fn max_adjustment(&self) -> Option<f64> {
        None
    }
}
