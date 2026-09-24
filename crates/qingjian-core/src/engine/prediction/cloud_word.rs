use crate::candidate::{Candidate, CandidateKind};

/// 云端给出的一个词候选：文本加全拼音节，音节用来校验它确实对得上用户敲的拼音，也用来记成用户词。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CloudWord {
    /// 词。
    pub text: String,

    /// 全拼音节，如 `["zhang", "tao"]`。问字模式的答案不校验拼音，这里为空。
    pub syllables: Vec<String>,

    /// 显示用读音（问字模式答案的带声调拼音，如 `sēn`）。
    pub reading: Option<String>,

    /// 本机模型给的（本地联想 / 同音纠错），不是云端：界面上不画云朵。
    pub local: bool,
}

impl CloudWord {
    /// 本机模型给出的一个词（本地联想 / 同音纠错用）。
    pub fn local(text: String, syllables: Vec<String>) -> Self {
        Self {
            text,
            syllables,
            reading: None,
            local: true,
        }
    }

    /// 转成候选（译文留给 `Engine::annotate` 补）；本机来源的标 `Local`，不画云朵。
    pub fn into_candidate(self) -> Candidate {
        Candidate {
            text: self.text,
            kind: if self.local {
                CandidateKind::Local
            } else {
                CandidateKind::Cloud
            },
            syllables: self.syllables,
            reading: self.reading,
            translation: None,
        }
    }
}
