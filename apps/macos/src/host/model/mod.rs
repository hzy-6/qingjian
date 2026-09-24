//! 本地整句模型：后台加载、停顿后请求重排、结果到了重画当前页。
//!
//! 按键回调里永远只跑词级模型；模型的意见在停键 80 毫秒后请求、一百毫秒上下后到，只换候选窗口里的整句候选，
//! 用户翻过页或动过高亮就不打扰。前文优先用应用里光标前的文字（`refresh` 每次查询前给 Engine），应用给不出退回本会话历史。

use std::sync::mpsc::{TryRecvError, channel};

use qingjian_core::sentence::SentenceScorer;

mod rescore_monitor;

pub(super) use rescore_monitor::RescoreMonitor;

use super::*;

/// 后台加载的结果：Qwen GGUF 模型，接 Engine 时统一成 trait 对象。
pub(super) type LoadedModel = Result<Box<dyn SentenceScorer>, String>;

impl Host {
    /// 在后台线程加载 Qwen GGUF 并预热（第一次前向要编译 Metal 内核，几百毫秒到几秒），加载完由
    /// [`Self::attach_loaded_model`] 接上。没有模型文件就什么都不做。
    pub(super) fn load_local_model(&mut self) {
        if self.model_loader.is_some() || self.engine.has_sentence_scorer() {
            return;
        }
        let Some(path) = paths::qwen_path() else {
            tracing::info!("没有本地整句模型文件，不重排");
            return;
        };
        let (tx, rx) = channel::<LoadedModel>();
        let spawned = std::thread::Builder::new()
            .name("qingjian-model-load".to_owned())
            .spawn(move || {
                let started = std::time::Instant::now();
                let model = qingjian_qwen::QwenScorer::load(&path)
                    .map(|scorer| Box::new(scorer) as Box<dyn SentenceScorer>)
                    .map_err(|error| error.to_string());
                // 按生产批量与常见长度预热：Metal 会按张量形状懒编译；单字单路径不足以覆盖真正的
                // 八路径长句，首次长句仍可能卡几秒。整个过程都在加载线程里，不阻塞按键。
                let warmup = [
                    "今天下午开会",
                    "今天下午开会吗",
                    "今天下午可以开会",
                    "今天下午我们开会",
                    "今天下午开始开会",
                    "今天下午开个会议",
                    "今天下午开会讨论",
                    "今天下午再开会吧",
                ];
                let model = model.and_then(|scorer| match scorer.score("前文", &warmup) {
                    scores if scores.len() == warmup.len() => Ok(scorer),
                    _ => Err("预热打分没有返回结果".to_owned()),
                });
                if model.is_ok() {
                    tracing::info!(
                        path = %path.display(),
                        total_ms = started.elapsed().as_millis(),
                        "本地整句模型已加载并预热"
                    );
                }
                let _ = tx.send(model);
            });
        match spawned {
            Ok(_) => self.model_loader = Some(rx),
            Err(error) => tracing::warn!(%error, "起不了模型加载线程，本地整句模型不用"),
        }
    }

    /// 加载线程有结果了就接到 Engine 上；每次查询顺手看一眼，不阻塞。
    pub fn attach_loaded_model(&mut self) {
        let Some(rx) = &self.model_loader else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(scorer)) => {
                self.engine.set_async_sentence_scorer(Some(scorer));
                self.model_loader = None;
                // 模型上线了：日志里补一条会话信息，之后的条目知道重排开着
                let version = self.version.clone();
                self.engine.log_session(&version, "macos");
            }
            Ok(Err(error)) => {
                tracing::warn!(%error, "本地整句模型加载失败，不重排");
                self.model_loader = None;
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => self.model_loader = None,
        }
    }

    /// 卸掉模型（配置关掉）。
    pub(super) fn unload_local_model(&mut self) {
        self.model_loader = None;
        self.engine.set_async_sentence_scorer(None);
        self.engine.set_character_proposer(None);
        self.rescore.stop();
    }

    /// 加载字符级整句提议模型（`char5.fst`，mmap，很快）：与 Qwen 重排一起启用，
    /// 重排池里除词级路径外再补同音字级候选。没有文件就跳过。
    pub(super) fn load_character_model(&mut self) {
        let Some(path) = paths::char_model_path() else {
            tracing::info!("没有字符级整句提议模型，不用");
            return;
        };
        match CharNgramModel::from_path(&path) {
            Ok(model) => {
                tracing::info!(path = %path.display(), "字符级整句提议已加载");
                self.engine.set_character_proposer(Some(Box::new(model)));
            }
            Err(error) => {
                tracing::warn!(%error, path = %path.display(), "字符级整句提议加载失败")
            }
        }
    }

    /// 每次查询之后：有整句路径等着打分就起防抖计时。
    pub fn schedule_rescoring(&mut self) {
        if self.engine.rescoring_pending() {
            self.rescore.schedule();
        }
    }

    /// 防抖到点：把攒着的整句路径送去后台，开始轮询。
    pub fn start_rescoring(&mut self) {
        if self.engine.composition().is_empty() {
            self.rescore.stop();
            return;
        }
        if self.engine.request_rescoring() {
            self.rescore.start_polling();
        }
    }

    /// 轮询到点：分回来了就重查一次、重画当前页；用户已翻页或动过高亮就只留着分不动画面。
    pub fn poll_rescoring(&mut self) {
        if self.engine.composition().is_empty() || self.translation.is_some() {
            self.rescore.stop();
            return;
        }
        if !self.engine.poll_rescoring() {
            // 等太久多半是前文变了（上屏后接着打下一段）、结果作废；真卡住也只是这轮不重排
            if self.rescore.expired() {
                tracing::debug!("等本地整句模型超时，本轮不重排");
                self.rescore.stop();
            }
            return;
        }
        self.rescore.stop();
        if self.session.page != 0 || self.session.navigated {
            return;
        }
        let Ok(mut query) = self.engine.query() else {
            return;
        };
        self.engine.annotate(&mut query.candidates);
        let preedit = Preedit::from_marked(&query.marked_segments(), query.marked_cursor());
        let cloud = self.session.layout.cloud().to_vec();
        self.reset_session(preedit, query.candidates.items);
        if !cloud.is_empty() {
            self.session.layout.set_cloud(cloud);
        }
        self.render();
    }
}
