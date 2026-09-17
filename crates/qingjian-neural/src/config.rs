use std::path::Path;

use serde::Deserialize;

use crate::NeuralError;

/// 模型结构，来自导出目录的 `config.json`（训练脚本 `common.py::ModelConfig` 原样写出）。
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// 字表大小。
    pub vocab_size: usize,

    /// 层数。
    pub n_layer: usize,

    /// 隐层宽度。
    pub n_embd: usize,

    /// 注意力头数。
    pub n_head: usize,

    /// 最长上下文（token 数），位置嵌入的行数。
    pub context: usize,

    /// 训练侧建议的单条路径最大神经修正（nat）：模型自带「该被信多少」，Engine 在用户没配置时用它。
    /// 旧模型没有这个字段，缺省 `None`（用 Core 的缺省上限），加载不因缺字段失败。
    #[serde(default)]
    pub max_adjustment: Option<f64>,
}

impl ModelConfig {
    /// 解析 `config.json` 的正文；`path` 只用来报错。
    pub(crate) fn from_json(text: &str, path: &Path) -> Result<Self, NeuralError> {
        serde_json::from_str(text).map_err(|source| NeuralError::Json {
            path: path.to_owned(),
            source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal() -> String {
        r#"{"vocab_size": 100, "n_layer": 2, "n_embd": 32, "n_head": 4, "context": 128}"#.to_owned()
    }

    /// 旧模型的 config.json 没有修正建议字段：加载成功，用 `None`（Core 的缺省上限）。
    #[test]
    fn parses_without_the_suggested_cap() {
        let cfg = ModelConfig::from_json(&minimal(), Path::new("config.json")).unwrap();
        assert_eq!(cfg.max_adjustment, None);
        assert_eq!(cfg.context, 128);
    }

    #[test]
    fn parses_the_suggested_cap() {
        let text = minimal().replace('}', ", \"max_adjustment\": 2.5}");
        let cfg = ModelConfig::from_json(&text, Path::new("config.json")).unwrap();
        assert_eq!(cfg.max_adjustment, Some(2.5));
    }
}
