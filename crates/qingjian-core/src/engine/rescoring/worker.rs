use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

use crate::sentence::SentenceScorer;

/// 一次打分任务：前文与一批文本。`sequence` 是引擎发出的单调递增请求序号，原样带回结果。
struct Job {
    sequence: u64,
    before: String,
    after: String,
    texts: Vec<String>,
}

/// 打好的分，与任务一一对应。`poll_rescoring` 按序号丢弃被更新请求顶掉的旧结果。
pub(crate) struct Scored {
    pub sequence: u64,
    pub before: String,
    pub after: String,
    pub texts: Vec<String>,
    pub scores: Vec<f64>,
}

/// 后台打分线程：模型前向要几十毫秒，不能放在按键回调里。
/// 任务排队时只算最新的一条（旧的对应已经过去的输入状态）；线程随本结构一起结束。
pub(crate) struct RescoreWorker {
    jobs: Sender<Job>,
    results: Receiver<Scored>,
    handle: Option<JoinHandle<()>>,
}

impl RescoreWorker {
    pub fn spawn(scorer: Box<dyn SentenceScorer>) -> Self {
        let (jobs, job_rx) = channel::<Job>();
        let (result_tx, results) = channel::<Scored>();
        let handle = std::thread::Builder::new()
            .name("qingjian-rescore".to_owned())
            .spawn(move || {
                while let Ok(mut job) = job_rx.recv() {
                    // 攒了好几条只算最后一条（它的序号最新）
                    while let Ok(newer) = job_rx.try_recv() {
                        job = newer;
                    }
                    let texts: Vec<&str> = job.texts.iter().map(String::as_str).collect();
                    let started = std::time::Instant::now();
                    let scores = scorer.score_with_after(&job.before, &job.after, &texts);
                    tracing::debug!(
                        texts = texts.len(),
                        context_chars = job.before.chars().count(),
                        ms = started.elapsed().as_millis(),
                        "神经重打分完成"
                    );
                    let done = Scored {
                        sequence: job.sequence,
                        before: job.before,
                        after: job.after,
                        texts: job.texts,
                        scores,
                    };
                    if result_tx.send(done).is_err() {
                        break;
                    }
                }
            })
            .ok();
        if handle.is_none() {
            tracing::warn!("起不了神经重打分线程，本次不用模型");
        }
        Self {
            jobs,
            results,
            handle,
        }
    }

    pub fn is_alive(&self) -> bool {
        // 线程真死过（打分器 panic 等）就当没有异步重排器：引擎退化成同步路径外的「不重排」，别让它
        // 永远报活着、壳每次停顿都白跑一趟请求
        self.handle
            .as_ref()
            .is_none_or(|handle| !handle.is_finished())
    }

    /// 提交一次打分任务。`sequence` 由引擎分配（单调递增），结果原样带回；相同 (before, after, texts)
    /// 的两次提交都真实执行——第二次往往就是为了淘汰上一次的旧结果。
    pub fn submit(&self, sequence: u64, before: String, after: String, texts: Vec<String>) {
        if self
            .jobs
            .send(Job {
                sequence,
                before,
                after,
                texts,
            })
            .is_err()
        {
            tracing::warn!("神经重打分线程已退出");
        }
    }

    /// 取一条打好的分；没有就 `None`。
    pub fn poll(&self) -> Option<Scored> {
        self.results.try_recv().ok()
    }
}

impl Drop for RescoreWorker {
    fn drop(&mut self) {
        // 关掉任务通道线程就会退出；不等它（模型可能正算到一半）
        let _ = self.handle.take();
    }
}
