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

/// Qwen GGUF（llama.cpp）：用户目录与包内 `model/` 目录里挑选规则与 bundle.sh 一致——
/// 裁剪过的 `*-text.gguf` 优先、同档按文件名取最小（`read_dir` 不保证顺序，显式排序）。
pub fn qwen_path() -> Option<PathBuf> {
    fn first_gguf(dir: &Path) -> Option<PathBuf> {
        let mut ggufs: Vec<PathBuf> = std::fs::read_dir(dir)
            .ok()?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "gguf"))
            .collect();
        ggufs.sort_by_key(|path| {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            // false 排在 true 前:裁剪版(-text)优先,再按文件名
            (!name.contains("-text."), name)
        });
        ggufs.into_iter().next()
    }
    // 用户目录建不了/读不了(HOME 缺失、磁盘满、iCloud 只读)不该连包内模型都放弃
    user_data_dir()
        .and_then(|dir| first_gguf(&dir.join("model")))
        .or_else(|| {
            resources_dir()
                .ok()
                .and_then(|dir| first_gguf(&dir.join("model")))
        })
}
