use qingjian_core::sentence::SentenceScorer;

use crate::CharScorer;

impl SentenceScorer for CharScorer {
    fn score(&self, context: &str, texts: &[&str]) -> Vec<f64> {
        match CharScorer::score(self, context, texts) {
            Ok(scores) => scores,
            Err(error) => {
                tracing::warn!(%error, "神经重打分失败，本次不用");
                Vec::new()
            }
        }
    }

    fn score_with_after(&self, before: &str, after: &str, texts: &[&str]) -> Vec<f64> {
        match CharScorer::score_with_after(self, before, after, texts) {
            Ok(scores) => scores,
            Err(error) => {
                tracing::warn!(%error, "神经双向重打分失败，本次不用");
                Vec::new()
            }
        }
    }

    /// `config.json` 里模型自带的修正建议（旧模型 `None`，用 Core 的缺省上限）；非法值也当没有。
    fn max_adjustment(&self) -> Option<f64> {
        self.suggested_max_adjustment()
    }
}
