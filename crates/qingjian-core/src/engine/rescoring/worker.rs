use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

use crate::sentence::SentenceScorer;

/// 后台线程里两类任务的优先级：重排决定用户看到的第一候选，预测只是补充槽位，
/// 两类都排队时先算重排。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JobKind {
    Rescore,
    Predict,
}

/// 一次打分任务：前文与一批文本。`sequence` 是发出方分配的单调递增请求序号，原样带回结果。
struct Job {
    kind: JobKind,
    sequence: u64,
    before: String,
    after: String,
    texts: Vec<String>,
}

/// 打好的分，与任务一一对应。重排与预测各走各的结果渠道，消费方互不抢。
pub(crate) struct Scored {
    pub sequence: u64,
    pub before: String,
    pub after: String,
    pub texts: Vec<String>,
    pub scores: Vec<f64>,
}

/// 后台打分线程：模型前向要几十毫秒，不能放在按键回调里。
/// 任务排队时同类只算最新的一条（旧的对应已经过去的输入状态）；重排永远先算。
/// 线程随本结构一起结束。
pub(crate) struct RescoreWorker {
    /// 关掉它线程的 `recv` 就会返回，线程收尾退出。
    jobs: Option<Sender<Job>>,
    rescore_results: Receiver<Scored>,
    predict_results: Receiver<Scored>,
    handle: Option<JoinHandle<()>>,
}

impl RescoreWorker {
    pub fn spawn(scorer: Box<dyn SentenceScorer>) -> Self {
        let (jobs, job_rx) = channel::<Job>();
        let (rescore_tx, rescore_results) = channel::<Scored>();
        let (predict_tx, predict_results) = channel::<Scored>();
        let handle = std::thread::Builder::new()
            .name("qingjian-rescore".to_owned())
            .spawn(move || {
                let run = |job: Job| -> Scored {
                    let Job {
                        sequence,
                        before,
                        after,
                        texts,
                        ..
                    } = job;
                    let texts_ref: Vec<&str> = texts.iter().map(String::as_str).collect();
                    let started = std::time::Instant::now();
                    let scores = scorer.score_with_after(&before, &after, &texts_ref);
                    tracing::debug!(
                        texts = texts_ref.len(),
                        context_chars = before.chars().count(),
                        ms = started.elapsed().as_millis(),
                        "神经打分完成"
                    );
                    Scored {
                        sequence,
                        before,
                        after,
                        texts,
                        scores,
                    }
                };
                while let Ok(first) = job_rx.recv() {
                    // 排队只留每类最新的一条；重排先算、预测后算
                    let mut rescore: Option<Job> = None;
                    let mut predict: Option<Job> = None;
                    let mut current = Some(first);
                    while let Some(job) = current.take() {
                        match job.kind {
                            JobKind::Rescore => rescore = Some(job),
                            JobKind::Predict => predict = Some(job),
                        }
                        match job_rx.try_recv() {
                            Ok(newer) => current = Some(newer),
                            Err(_) => break,
                        }
                    }
                    if let Some(job) = rescore
                        && rescore_tx.send(run(job)).is_err()
                    {
                        break;
                    }
                    if let Some(job) = predict
                        && predict_tx.send(run(job)).is_err()
                    {
                        break;
                    }
                }
            })
            .ok();
        if handle.is_none() {
            tracing::warn!("起不了神经重打分线程，本次不用模型");
        }
        Self {
            jobs: Some(jobs),
            rescore_results,
            predict_results,
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

    /// 提交一次重排任务。`sequence` 由引擎分配（单调递增），结果原样带回；相同 (before, after, texts)
    /// 的两次提交都真实执行——第二次往往就是为了淘汰上一次的旧结果。
    pub fn submit(&self, sequence: u64, before: String, after: String, texts: Vec<String>) {
        self.send(Job {
            kind: JobKind::Rescore,
            sequence,
            before,
            after,
            texts,
        });
    }

    /// 提交一次本地联想打分（一批接续提议）。与重排共用模型实例，排队时重排优先。
    pub fn submit_predict(&self, sequence: u64, before: String, after: String, texts: Vec<String>) {
        self.send(Job {
            kind: JobKind::Predict,
            sequence,
            before,
            after,
            texts,
        });
    }

    fn send(&self, job: Job) {
        let Some(jobs) = &self.jobs else {
            return;
        };
        if jobs.send(job).is_err() {
            tracing::warn!("神经打分线程已退出");
        }
    }

    /// 取一条重排打好的分；没有就 `None`。
    pub fn poll(&self) -> Option<Scored> {
        self.rescore_results.try_recv().ok()
    }

    /// 取一条本地联想打好的分；没有就 `None`。
    pub fn poll_predict(&self) -> Option<Scored> {
        self.predict_results.try_recv().ok()
    }
}

impl Drop for RescoreWorker {
    fn drop(&mut self) {
        // 先关任务通道（线程的 recv 返回、算完手上这条就退出），再等它真正退完。
        // **join 不能省**：模型（与它的 Metal 资源集）活在那个线程里，不等它退完就继续析构，
        // 进程退出时 ggml 会撞上 `GGML_ASSERT([rsets->data count] == 0)` 直接 abort。
        // 手上最多一条几十到几百毫秒的推理，卸载 / 退出时等一下可以接受。
        self.jobs = None;
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
