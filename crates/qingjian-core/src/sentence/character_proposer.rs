//! 拼音约束的字符级整句提议接口；具体语言模型在 `qingjian-lm`。

/// 对每个音节给定一组同音字，产出按模型分数降序的完整句子。
pub trait CharacterProposer: Send + Sync {
    fn propose(&self, choices: &[Vec<char>], beam_width: usize, limit: usize)
    -> Vec<(String, f64)>;

    fn score_text(&self, text: &str) -> f64;
}
