use std::time::{Duration, Instant};

use super::*;
use crate::sentence::SentenceWord;

/// 假打分器：偏爱某个文本，其余都给低分。
struct Prefers(&'static str);

impl SentenceScorer for Prefers {
    fn score(&self, _context: &str, texts: &[&str]) -> Vec<f64> {
        texts
            .iter()
            .map(|t| if *t == self.0 { -1.0 } else { -20.0 })
            .collect()
    }
}

/// 每次调用偏爱不同文本的打分器（第一次 开放、第二次 开饭……），并数调用次数、慢一拍；
/// 用来证明相同请求真的执行了、旧请求的迟到结果不会再被收下。
struct Alternating {
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,

    /// 每次打分前睡多久（毫秒），0 不睡。
    delay_ms: u64,
}

impl SentenceScorer for Alternating {
    fn score(&self, _context: &str, texts: &[&str]) -> Vec<f64> {
        let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        if self.delay_ms > 0 {
            std::thread::sleep(Duration::from_millis(self.delay_ms));
        }
        let preferred = if call % 2 == 1 { "开放" } else { "开饭" };
        texts
            .iter()
            .map(|t| if *t == preferred { -1.0 } else { -20.0 })
            .collect()
    }
}

/// 记下每一批送去打分的文本，分数全给 -1。
struct RecordsBatches(std::sync::Arc<std::sync::Mutex<Vec<Vec<String>>>>);

impl SentenceScorer for RecordsBatches {
    fn score(&self, _context: &str, texts: &[&str]) -> Vec<f64> {
        self.0
            .lock()
            .unwrap()
            .push(texts.iter().map(|t| (*t).to_owned()).collect());
        texts.iter().map(|_| -1.0).collect()
    }
}

/// 所有分数都给同一个非有限值（NaN / ±inf），模拟模型出异常。
struct Broken(f64);

impl SentenceScorer for Broken {
    fn score(&self, _context: &str, texts: &[&str]) -> Vec<f64> {
        vec![self.0; texts.len()]
    }
}

/// 行为同 [`Prefers`]，但声明模型建议的修正上限。
struct SuggestsCap {
    preferred: &'static str,
    cap: f64,
}

impl SentenceScorer for SuggestsCap {
    fn score(&self, _context: &str, texts: &[&str]) -> Vec<f64> {
        texts
            .iter()
            .map(|t| if *t == self.preferred { -1.0 } else { -20.0 })
            .collect()
    }

    fn max_adjustment(&self) -> Option<f64> {
        Some(self.cap)
    }
}

struct RecordsAfter(std::sync::Arc<std::sync::Mutex<String>>);

impl SentenceScorer for RecordsAfter {
    fn score(&self, _context: &str, texts: &[&str]) -> Vec<f64> {
        texts.iter().map(|_| -1.0).collect()
    }

    fn score_with_after(&self, before: &str, after: &str, texts: &[&str]) -> Vec<f64> {
        *self.0.lock().unwrap() = format!("{before}|{after}");
        texts.iter().map(|_| -1.0).collect()
    }
}

fn path(text: &str, score: f64) -> Conversion {
    personal_path(text, score, score)
}

/// 带静态分的路径：`score − static_score` 是个人证据（个人 n-gram、用户加分、代价）。
fn personal_path(text: &str, score: f64, static_score: f64) -> Conversion {
    Conversion {
        text: text.to_owned(),
        syllables: Vec::new(),
        words: vec![SentenceWord {
            text: text.to_owned(),
            syllables: Vec::new(),
            placeholder: false,
        }],
        score,
        static_score,
        penalty: 0.0,
    }
}

fn engine() -> Engine {
    Engine::new(Dictionary::parse("开发\tkai fa\t9000\n").unwrap())
}

fn texts(paths: &[Conversion]) -> Vec<&str> {
    paths.iter().map(|p| p.text.as_str()).collect()
}

/// 轮询到后台打分结果回来并进了缓存。
fn wait_for_rescoring(engine: &mut Engine) {
    let started = Instant::now();
    while !engine.poll_rescoring() {
        assert!(started.elapsed() < Duration::from_secs(5), "后台没回结果");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// 等后台打分真正开始（进了打分器）这么多次。
fn wait_for_calls(calls: &std::sync::atomic::AtomicUsize, want: usize) {
    let started = Instant::now();
    while calls.load(std::sync::atomic::Ordering::SeqCst) < want {
        assert!(started.elapsed() < Duration::from_secs(5), "后台打分没开始");
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn sync_scorer_reorders_paths_in_place() {
    let engine = engine().with_sentence_scorer(Box::new(Prefers("开放")), Some(0.5), None, None);
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert_eq!(texts(&paths), ["开放", "开饭"]);
    // λ 0.5：开饭 −10 + 0.5·(−20 + 10) = −15；开放 −11 + 0.5·(−1 + 11) = −6
    assert!((paths[0].score - -6.0).abs() < 1e-9);
    assert!((paths[1].score - -15.0).abs() < 1e-9);
    assert!(!engine.rescoring_pending());
}

#[test]
fn sync_scorer_receives_surrounding_after_text() {
    let seen = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let mut engine =
        engine().with_sentence_scorer(Box::new(RecordsAfter(seen.clone())), Some(0.5), None, None);
    engine.set_rescoring_surrounding(Some("前文".into()), Some("后文".into()));
    let mut paths = vec![path("开发", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert_eq!(&*seen.lock().unwrap(), "前文|后文");
}

#[test]
fn async_scorer_waits_for_the_shell_to_request_and_poll() {
    let mut engine =
        engine().with_async_sentence_scorer(Box::new(Prefers("开放")), Some(0.5), None, None);
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    // 第一次：没分，顺序不动，记下要分的
    assert_eq!(texts(&paths), ["开饭", "开放"]);
    assert!(engine.rescoring_pending());
    assert!(engine.request_rescoring());
    assert!(!engine.rescoring_pending());
    let started = Instant::now();
    while !engine.poll_rescoring() {
        assert!(started.elapsed() < Duration::from_secs(5), "后台没回结果");
        std::thread::sleep(Duration::from_millis(5));
    }
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert_eq!(texts(&paths), ["开放", "开饭"]);
    // 没有新的要打的就不发
    assert!(!engine.request_rescoring());
}

#[test]
fn a_changed_context_discards_the_cached_scores() {
    let mut engine =
        engine().with_async_sentence_scorer(Box::new(Prefers("开放")), Some(0.5), None, None);
    engine.history_mut().record("今天");
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert!(engine.request_rescoring());
    let started = Instant::now();
    while !engine.poll_rescoring() {
        assert!(started.elapsed() < Duration::from_secs(5));
        std::thread::sleep(Duration::from_millis(5));
    }
    // 上屏了别的字，前文变了：缓存作废，又得重新要
    engine.history_mut().record("很好");
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert_eq!(texts(&paths), ["开饭", "开放"]);
    assert!(engine.rescoring_pending());
}

#[test]
fn a_changed_after_context_discards_the_cached_scores() {
    let mut engine =
        engine().with_async_sentence_scorer(Box::new(Prefers("开放")), Some(0.5), None, None);
    engine.set_rescoring_surrounding(Some("前文".into()), Some("后文一".into()));
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert!(engine.request_rescoring());
    let started = Instant::now();
    while !engine.poll_rescoring() {
        assert!(started.elapsed() < Duration::from_secs(5));
        std::thread::sleep(Duration::from_millis(5));
    }
    engine.set_rescoring_surrounding(Some("前文".into()), Some("后文二".into()));
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert!(engine.rescoring_pending());
    // 前文相同、后文不同：候选分数按新的后文重新打一遍，重排生效
    assert!(engine.request_rescoring());
    wait_for_rescoring(&mut engine);
    engine.rescore_paths(&mut paths);
    assert_eq!(texts(&paths), ["开放", "开饭"]);
}

#[test]
fn the_shell_context_wins_over_session_history() {
    let mut engine = engine().with_sentence_scorer(Box::new(Prefers("开放")), None, None, Some(4));
    engine.history_mut().record("本会话上屏的历史");
    assert_eq!(engine.rescoring_context(), "屏的历史");
    engine.set_rescoring_context(Some("应用里光标前的文本".to_owned()));
    assert_eq!(engine.rescoring_context(), "前的文本");
    engine.set_rescoring_context(None);
    assert_eq!(engine.rescoring_context(), "屏的历史");
}

#[test]
fn the_same_text_is_scored_once_per_context() {
    let batches = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut engine = engine().with_sentence_scorer(
        Box::new(RecordsBatches(batches.clone())),
        Some(0.5),
        None,
        None,
    );
    engine.set_rescoring_surrounding(Some("前文".into()), None);
    // 相邻拼音的第一键：两条路径都要打分
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    // 多敲一键的下一键：开饭 还在（缓存命中），只有新出来的 开饭馆 要打
    let mut paths = vec![path("开饭", -10.0), path("开饭馆", -11.5)];
    engine.rescore_paths(&mut paths);
    assert_eq!(
        *batches.lock().unwrap(),
        vec![
            vec!["开饭".to_owned(), "开放".to_owned()],
            vec!["开饭馆".to_owned()],
        ]
    );
}

#[test]
fn emptying_the_composition_invalidates_the_cache() {
    let mut engine =
        engine().with_async_sentence_scorer(Box::new(Prefers("开放")), Some(0.5), None, None);
    engine.set_rescoring_surrounding(Some("前文".into()), Some("后文".into()));
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert!(engine.request_rescoring());
    wait_for_rescoring(&mut engine);
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert_eq!(texts(&paths), ["开放", "开饭"]);
    // 删空缓冲区：同一段前文下已经算出的分不再有效，同文本要重新问模型
    engine.push('k');
    assert!(engine.backspace());
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert_eq!(texts(&paths), ["开饭", "开放"]);
    assert!(engine.rescoring_pending());
}

#[test]
fn identical_requests_after_invalidation_are_rescored() {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut engine = engine().with_async_sentence_scorer(
        Box::new(Alternating {
            calls: calls.clone(),
            delay_ms: 0,
        }),
        Some(1.0),
        None,
        None,
    );
    engine.set_rescoring_surrounding(Some("前文".into()), Some("后文".into()));
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert!(engine.request_rescoring());
    wait_for_rescoring(&mut engine);
    engine.rescore_paths(&mut paths);
    // 第一次调用偏爱 开放
    assert_eq!(texts(&paths), ["开放", "开饭"]);
    // 删空再重来：同样的 (before, after, texts) 第二次请求必须真实执行（淘汰旧结果），不能按内容去重
    engine.push('k');
    assert!(engine.backspace());
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert!(engine.request_rescoring());
    wait_for_rescoring(&mut engine);
    engine.rescore_paths(&mut paths);
    // 第二次调用偏爱 开饭：分确实重算过
    assert_eq!(texts(&paths), ["开饭", "开放"]);
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
}

#[test]
fn late_results_from_superseded_requests_are_dropped() {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut engine = engine().with_async_sentence_scorer(
        Box::new(Alternating {
            calls: calls.clone(),
            delay_ms: 40,
        }),
        Some(0.5),
        None,
        None,
    );
    engine.set_rescoring_surrounding(Some("前文".into()), Some("后文甲".into()));
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert!(engine.request_rescoring());
    // 第一条任务真正开始算了，第二条才不会被后台线程合并掉
    wait_for_calls(&calls, 1);
    engine.set_rescoring_surrounding(Some("前文".into()), Some("后文乙".into()));
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert!(engine.request_rescoring());
    wait_for_calls(&calls, 2);
    // 上下文又回到 后文甲：先查一次让缓存的前文键也换回去，第一条的迟到结果上下文字符串
    // 对得上（ABA），但序号已被第二条顶掉，一个都不能收
    engine.set_rescoring_surrounding(Some("前文".into()), Some("后文甲".into()));
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert!(!engine.poll_rescoring());
    // 没收到分：顺序还是原样，等着重新打分
    assert_eq!(texts(&paths), ["开饭", "开放"]);
    assert!(engine.rescoring_pending());
}

#[test]
fn invalidation_rejects_old_result_before_a_new_request() {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut engine = engine().with_async_sentence_scorer(
        Box::new(Alternating {
            calls: calls.clone(),
            delay_ms: 40,
        }),
        Some(0.5),
        None,
        None,
    );
    engine.set_rescoring_surrounding(Some("前文".into()), Some("后文".into()));
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert!(engine.request_rescoring());
    wait_for_calls(&calls, 1);
    // 删空后立刻重输，缓存键又回到原值；新请求尚未发出。
    engine.push('k');
    assert!(engine.backspace());
    engine.rescore_paths(&mut paths);
    std::thread::sleep(Duration::from_millis(60));
    assert!(!engine.poll_rescoring(), "删空前的结果不能进入新组句的缓存");
    assert!(engine.rescoring_pending());
}

#[test]
fn nonfinite_scores_keep_the_static_order() {
    for broken in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let engine = engine().with_sentence_scorer(Box::new(Broken(broken)), Some(0.5), None, None);
        let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
        engine.rescore_paths(&mut paths);
        // 异常分数不参与调整：顺序与分数都和没接模型时一致
        assert_eq!(texts(&paths), ["开饭", "开放"], "{broken}");
        assert!((paths[0].score - -10.0).abs() < 1e-9, "{broken}");
        assert!((paths[1].score - -11.0).abs() < 1e-9, "{broken}");
    }
}

/// 池大小可配：k 调大后，原本进不了池的深名次路径也能被重排拉上来。
#[test]
fn a_larger_path_pool_lets_deep_paths_win() {
    /// 给每条文本一个固定分的打分器:偏爱文本之外的按字典序打分,便于构造深名次翻案
    struct Ranking(std::collections::HashMap<String, f64>);
    impl SentenceScorer for Ranking {
        fn score(&self, _context: &str, texts: &[&str]) -> Vec<f64> {
            texts
                .iter()
                .map(|t| *self.0.get(*t).unwrap_or(&-30.0))
                .collect()
        }
    }
    let mut scores = std::collections::HashMap::new();
    scores.insert("开饭".to_owned(), -20.0);
    scores.insert("开放".to_owned(), -1.0);
    let mut engine = engine().with_sentence_scorer(
        Box::new(Ranking(scores)),
        Some(1.0),
        // margin 放全:别在打分前把深名次路径删掉
        Some(100.0),
        None,
    );
    engine.set_neural_paths(2);
    // 词库里 开发 一枝独秀,开饭/开放 都不是常用路径;两条路径分接近,深名次的 开放 靠神经分翻上来
    let mut paths = vec![
        personal_path("开饭", -10.0, -10.0),
        personal_path("开放", -11.0, -11.0),
    ];
    engine.rescore_paths(&mut paths);
    assert_eq!(texts(&paths), ["开放", "开饭"]);
    assert_eq!(engine.neural_paths(), 2);
    engine.set_neural_paths(0);
    assert_eq!(engine.neural_paths(), 1, "k 钳底到 1");
}

/// 闸：守成路径有个人证据优势时，神经分差追不上 gate 倍就不许翻案。
#[test]
fn the_gate_blocks_a_flip_backed_by_weaker_neural_than_personal_evidence() {
    struct Mildly(&'static str);
    impl SentenceScorer for Mildly {
        fn score(&self, _context: &str, texts: &[&str]) -> Vec<f64> {
            texts
                .iter()
                .map(|t| if *t == self.0 { -4.0 } else { -9.0 })
                .collect()
        }
    }
    // 开饭 路径分 −10、静态 −13（个人证据 3 nat），开放 路径分 −11、静态 −11（没有）。
    // 神经偏爱 开放（−4 对 −9）：λ 0.5 下神经修正差 (−4+11)·0.5 − (−9+13)·0.5 = 2.5 nat，
    // 足以翻案（开放 −7.5 > 开饭 −8）但小于 3 nat 的个人证据优势
    let gated = || {
        let mut engine =
            engine().with_sentence_scorer(Box::new(Mildly("开放")), Some(0.5), None, None);
        engine.set_neural_gate(1.0);
        engine
    };
    // 不设闸的对照：闸的缺省已是 2，显式归零
    let mut ungated =
        engine().with_sentence_scorer(Box::new(Mildly("开放")), Some(0.5), None, None);
    ungated.set_neural_gate(0.0);
    let mut paths = vec![
        personal_path("开饭", -10.0, -13.0),
        personal_path("开放", -11.0, -11.0),
    ];
    ungated.rescore_paths(&mut paths);
    assert_eq!(
        texts(&paths),
        ["开放", "开饭"],
        "不设闸时 2.5 nat 的神经差翻得了案"
    );
    let mut paths = vec![
        personal_path("开饭", -10.0, -13.0),
        personal_path("开放", -11.0, -11.0),
    ];
    gated().rescore_paths(&mut paths);
    assert_eq!(
        texts(&paths),
        ["开饭", "开放"],
        "2.5 < 1×3，闸压回平手，守成者留前"
    );
    // 神经优势远超个人证据时闸拦不住（Prefers 的分差有 17 nat）
    let mut strong =
        engine().with_sentence_scorer(Box::new(Prefers("开放")), Some(1.0), None, None);
    strong.set_neural_gate(1.0);
    let mut paths = vec![
        personal_path("开饭", -10.0, -13.0),
        personal_path("开放", -11.0, -11.0),
    ];
    strong.rescore_paths(&mut paths);
    assert_eq!(
        texts(&paths),
        ["开放", "开饭"],
        "17 nat 的神经差盖过 3 nat 的个人证据"
    );
    // 挑战者自己带着更多个人证据时闸不拦（守成者的个人优势是负数）
    let mut reversed =
        engine().with_sentence_scorer(Box::new(Mildly("开饭")), Some(0.5), None, None);
    reversed.set_neural_gate(1.0);
    let mut paths = vec![
        personal_path("开放", -10.0, -10.0),
        personal_path("开饭", -11.0, -14.0),
    ];
    reversed.rescore_paths(&mut paths);
    assert_eq!(texts(&paths), ["开饭", "开放"]);
    // 非法值当 0(显式关闭)
    let mut invalid = engine();
    invalid.set_neural_gate(f64::NAN);
    assert_eq!(invalid.neural_gate(), 0.0);
    invalid.set_neural_gate(-3.0);
    assert_eq!(invalid.neural_gate(), 0.0);
    invalid.set_neural_gate(f64::INFINITY);
    assert_eq!(invalid.neural_gate(), 0.0);
}

/// 三条路径的链式压回:闸按下标(老排名)字典序处理、只向下压。测试锁定该遍历序的当前行为。
#[test]
fn the_gate_chains_downward_through_three_paths() {
    /// 按文本查固定分的打分器:开饭 / 开放 / 开工 各一个。
    struct Fixed([f64; 3]);
    impl SentenceScorer for Fixed {
        fn score(&self, _context: &str, texts: &[&str]) -> Vec<f64> {
            texts
                .iter()
                .map(|t| {
                    self.0[match *t {
                        "开饭" => 0,
                        "开放" => 1,
                        _ => 2,
                    }]
                })
                .collect()
        }
    }
    // 开饭 守成(路径 −10、静态 −13,个人证据 3),开放(−11/−11,无证据),开工(−12/−12,无证据)。
    // 神经 −9/−4/−20、λ0.5:开放对开饭的神经差 (−4+11)·0.5 − (−9+13)·0.5 = 1.5 < 2×3,全链压回,守成者第一。
    let mut gated =
        engine().with_sentence_scorer(Box::new(Fixed([-9.0, -4.0, -20.0])), Some(0.5), None, None);
    gated.set_neural_gate(2.0);
    let mut paths = vec![
        personal_path("开饭", -10.0, -13.0),
        personal_path("开放", -11.0, -11.0),
        personal_path("开工", -12.0, -12.0),
    ];
    gated.rescore_paths(&mut paths);
    assert_eq!(
        texts(&paths),
        ["开饭", "开放", "开工"],
        "1.5 nat 神经差翻不过 2×3 的闸"
    );
    // 神经差抬到 8(−16/0/−20):修正 开放 +5.5、开饭 −1.5,差 7 > 6,放行翻案
    let mut strong =
        engine().with_sentence_scorer(Box::new(Fixed([-16.0, 0.0, -20.0])), Some(0.5), None, None);
    strong.set_neural_gate(2.0);
    let mut paths = vec![
        personal_path("开饭", -10.0, -13.0),
        personal_path("开放", -11.0, -11.0),
        personal_path("开工", -12.0, -12.0),
    ];
    strong.rescore_paths(&mut paths);
    assert_eq!(
        texts(&paths),
        ["开放", "开饭", "开工"],
        "7 nat 神经差盖过 2×3 的闸"
    );
}

#[test]
fn a_smaller_max_adjustment_limits_the_reordering() {
    let mut clamped =
        engine().with_sentence_scorer(Box::new(Prefers("开放")), Some(1.0), None, None);
    clamped.set_neural_max_adjustment(Some(0.5));
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    clamped.rescore_paths(&mut paths);
    // ±10 的原始修正被压到 ±0.5，两条打平，稳定排序保持原顺序
    assert_eq!(texts(&paths), ["开饭", "开放"]);
    assert!((paths[0].score - -10.5).abs() < 1e-9);
    assert!((paths[1].score - -10.5).abs() < 1e-9);
    // 对照：缺省上限 8 下同样一批分数会翻盘
    let unclamped = engine().with_sentence_scorer(Box::new(Prefers("开放")), Some(1.0), None, None);
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    unclamped.rescore_paths(&mut paths);
    assert_eq!(texts(&paths), ["开放", "开饭"]);
}

#[test]
fn the_models_suggested_cap_applies_without_user_config() {
    let mut engine = engine().with_sentence_scorer(
        Box::new(SuggestsCap {
            preferred: "开放",
            cap: 1.0,
        }),
        Some(1.0),
        None,
        None,
    );
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    // 用户没配，用模型建议的 ±1：开饭 −10−1、开放 −11+1
    assert_eq!(texts(&paths), ["开放", "开饭"]);
    assert!((paths[0].score - -10.0).abs() < 1e-9);
    assert!((paths[1].score - -11.0).abs() < 1e-9);
    // 用户配置优先于模型建议；非法值（NaN）当没配，仍用模型建议
    engine.set_neural_max_adjustment(Some(0.25));
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert_eq!(texts(&paths), ["开饭", "开放"]);
    assert!((paths[0].score - -10.25).abs() < 1e-9);
    assert!((paths[1].score - -10.75).abs() < 1e-9);
    engine.set_neural_max_adjustment(Some(f64::NAN));
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert_eq!(texts(&paths), ["开放", "开饭"]);
}

#[test]
fn an_empty_after_reaches_the_scorer_on_both_paths() {
    let seen = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let mut synced =
        engine().with_sentence_scorer(Box::new(RecordsAfter(seen.clone())), Some(0.5), None, None);
    synced.set_rescoring_surrounding(Some("前文".into()), None);
    let mut paths = vec![path("开发", -10.0), path("开放", -11.0)];
    synced.rescore_paths(&mut paths);
    assert_eq!(&*seen.lock().unwrap(), "前文|");
    // 显式空串同 None 一样
    synced.set_rescoring_surrounding(Some("前文".into()), Some(String::new()));
    let mut paths = vec![path("开发", -10.0), path("开放", -11.0)];
    synced.rescore_paths(&mut paths);
    assert_eq!(&*seen.lock().unwrap(), "前文|");
    // 异步路径同样收到空后文
    let seen = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let mut engine = engine().with_async_sentence_scorer(
        Box::new(RecordsAfter(seen.clone())),
        Some(0.5),
        None,
        None,
    );
    engine.set_rescoring_surrounding(Some("前文".into()), None);
    let mut paths = vec![path("开发", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert!(engine.request_rescoring());
    wait_for_rescoring(&mut engine);
    assert_eq!(&*seen.lock().unwrap(), "前文|");
}

#[test]
fn a_legacy_scorer_ignores_the_after_context_on_the_async_path() {
    // 只实现 score 的旧模型（trait 默认忽略光标后文）在异步路径下照常工作
    let mut engine =
        engine().with_async_sentence_scorer(Box::new(Prefers("开放")), Some(0.5), None, None);
    engine.set_rescoring_surrounding(Some("前文".into()), Some("后文".into()));
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert!(engine.request_rescoring());
    wait_for_rescoring(&mut engine);
    let mut paths = vec![path("开饭", -10.0), path("开放", -11.0)];
    engine.rescore_paths(&mut paths);
    assert_eq!(texts(&paths), ["开放", "开饭"]);
}
