//! macOS 输入法与 CLI 共用的配置类型和附加词库加载。

mod config;
mod error;
pub mod extra_dictionaries;

pub use config::{
    AppsConfig, CandidateRenderer, Config, DEFAULT_DOMAINS, DEFAULT_ENGLISH_CANDIDATES_OFF,
    DEFAULT_ENGLISH_CANDIDATES_OFF_MACOS, DEFAULT_PAGE_KEYS, DictionariesConfig, GeneralConfig,
    KeyCombo, LEARNING_LANGUAGE_OFF, LayoutMode, LocalModelConfig, LogLevel, MAX_PAGE_SIZE,
    Modifiers, PAGE_KEY_OPTIONS, PreeditMode, ShortcutConfig, ThemeMode,
};
pub use error::ConfigError;
