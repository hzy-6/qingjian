/// 整句路径的第二打分来源：给「前文 + 整句文本」按字打 log 概率的模型（字级 Transformer，实现在 `qingjian-neural`）。
/// Core 只认这个 trait；Viterbi 出的前几条路径用它重打分，与路径本身的得分对数线性插值。
pub trait SentenceScorer: Send {
    /// 每条 `texts` 接在 `context`（光标前文，可空）后面的 `log P(text | context)`，与 `texts` 一一对应。
    /// 算不了（模型出错）返回空 Vec，调用方就当没有这个打分。
    fn score(&self, context: &str, texts: &[&str]) -> Vec<f64>;

    /// 可选的双向上下文入口。旧模型默认忽略光标后的文字，保持与旧 `.qjm` 模型兼容；
    /// 支持双向上下文的新模型可以覆盖此方法。
    fn score_with_after(&self, before: &str, _after: &str, texts: &[&str]) -> Vec<f64> {
        self.score(before, texts)
    }

    /// 模型自己建议的单条路径最大神经修正（nat）：训练侧知道该被信任多少，写在模型文件里。
    /// 旧模型没有这个字段，默认 `None`，用引擎的缺省上限 [`crate::NEURAL_MAX_ADJUSTMENT`]。
    fn max_adjustment(&self) -> Option<f64> {
        None
    }
}
