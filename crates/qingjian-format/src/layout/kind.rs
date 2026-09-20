/// 文件里装的是哪种数据。编号写进文件头，不要改已有编号。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum Kind {
    /// 拼音词库（`qingjian-dictionary::Dictionary`）。
    Dictionary = 1,

    /// 词级 bigram 语言模型（`qingjian-lm::BigramModel`）。
    LanguageModel = 2,

    /// 释义表。
    Glossary = 3,

    /// emoji 表。
    Emoji = 4,

    /// 英文词表。
    WordList = 5,
}

impl Kind {
    pub fn from_code(code: u16) -> Option<Self> {
        Some(match code {
            1 => Self::Dictionary,
            2 => Self::LanguageModel,
            3 => Self::Glossary,
            4 => Self::Emoji,
            5 => Self::WordList,
            // 6 曾是本地整句模型 .qjm:随包模型已换 GGUF,旧文件按「不认识的数据种类」处理
            _ => return None,
        })
    }

    pub fn code(self) -> u16 {
        self as u16
    }
}
