use serde::{Deserialize, Serialize};

/// 配置文件 `[model]` 分节：本地整句模型的开关与重排幅度。
///
/// 随包的字级小模型在本机给整句候选重新排序，全程离线、不联网，与云联想互不影响（本地先出、云端到了另占它自己的格）。
/// 模型文件不在包里（或用户目录 `model/` 里）时开关无效。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LocalModelConfig {
    /// 开着就加载模型、给整句重排。
    pub enabled: bool,

    /// 模型重排一条整句候选时分数最多挪动多少（nat）：模型与词库统计的分歧再大也只挪这么多，
    /// 调小重排更保守。`None`（缺省）跟随模型文件自带的建议，再退引擎的缺省值。
    pub max_adjustment: Option<f64>,
}

impl Default for LocalModelConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_adjustment: None,
        }
    }
}
