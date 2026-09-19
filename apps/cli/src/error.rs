use qingjian_dictionary::DictionaryError;
use qingjian_learning::LearningError;
use qingjian_lm::LmError;
use qingjian_neural::NeuralError;
use qingjian_platform::ConfigError;
use qingjian_predict::PredictError;
use qingjian_translate::GlossaryError;
#[derive(Debug, thiserror::Error)]
pub enum CliError {
    #[error(transparent)]
    Dictionary(#[from] DictionaryError),

    #[error(transparent)]
    Neural(#[from] NeuralError),

    #[error(transparent)]
    Glossary(#[from] GlossaryError),

    #[error(transparent)]
    Learning(#[from] LearningError),

    /// 学习语言不是 en / ja / es。
    #[error("learning language must be en, ja or es, got {0:?}")]
    Language(String),

    /// Qwen 小模型（`--qwen`）加载失败。
    #[error("qwen scorer: {0}")]
    Qwen(String),

    #[error(transparent)]
    Config(#[from] ConfigError),

    #[error(transparent)]
    Predict(#[from] PredictError),

    #[error(transparent)]
    LanguageModel(#[from] LmError),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Replay(#[from] crate::replay::ReplayError),

    #[error(transparent)]
    Eval(#[from] crate::eval::EvalError),

    #[error(transparent)]
    Tune(#[from] crate::tuning::TuneError),
}
