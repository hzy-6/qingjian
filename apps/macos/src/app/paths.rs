//! 数据文件位置：只读数据在 `.app/Contents/Resources/`，用户数据在 `~/Library/Application Support/Qingjian/`。

use std::path::{Path, PathBuf};

use objc2_foundation::NSBundle;

use crate::error::HostError;

/// 主 bundle 的 Resources 目录。
pub fn resources_dir() -> Result<PathBuf, HostError> {
    NSBundle::mainBundle()
        .resourcePath()
        .map(|p| PathBuf::from(p.to_string()))
        .ok_or(HostError::NoResources)
}

/// 某个资源文件的完整路径，不存在时报错而不是等到读取时才炸。
pub fn resource(name: &str) -> Result<PathBuf, HostError> {
    let path = resources_dir()?.join(name);
    if path.is_file() {
        Ok(path)
    } else {
        Err(HostError::MissingResource(path))
    }
}

/// 随包的领域词库目录：`.app/Contents/Resources/dicts/`；包里没有就是 `None`。
pub fn bundled_dicts_dir() -> Option<PathBuf> {
    let dir = resources_dir().ok()?.join("dicts");
    dir.is_dir().then_some(dir)
}

/// 配置文件：`~/Library/Application Support/Qingjian/config.toml`。
pub fn config_file() -> Option<PathBuf> {
    user_data_dir().map(|dir| dir.join("config.toml"))
}

/// 用户数据目录，不存在则创建。
pub fn user_data_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("HOME")?).join("Library/Application Support/Qingjian");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// 附加词库目录：`~/Library/Application Support/Qingjian/dicts/`，不存在则创建。
pub fn dicts_dir() -> Option<PathBuf> {
    let dir = user_data_dir()?.join("dicts");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// 本地整句模型（`.qjm` 单文件，或开发时的三件套目录）：
/// 用户目录 `model/` 里有就用它（自己训的），否则用包里的 `Resources/model/`；都没有是 `None`。
pub fn model_path() -> Option<PathBuf> {
    let user = user_data_dir()?.join("model");
    if let Some(found) = qingjian_neural::find_model(&user) {
        return Some(found);
    }
    qingjian_neural::find_model(&resources_dir().ok()?.join("model"))
}

/// Qwen GGUF（llama.cpp）：查找规则同 [`model_path`]，目录里有几份按文件名取最小的一份
/// （`read_dir` 不保证顺序，显式取 min 与 bundle.sh 的字典序一致）。有 GGUF 就用它重排（首选），没有再退回 `.qjm` 字级模型。
pub fn qwen_path() -> Option<PathBuf> {
    fn first_gguf(dir: &Path) -> Option<PathBuf> {
        std::fs::read_dir(dir)
            .ok()?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "gguf"))
            .min_by_key(|path| path.file_name().map(std::ffi::OsStr::to_owned))
    }
    first_gguf(&user_data_dir()?.join("model")).or_else(|| {
        resources_dir()
            .ok()
            .and_then(|dir| first_gguf(&dir.join("model")))
    })
}
