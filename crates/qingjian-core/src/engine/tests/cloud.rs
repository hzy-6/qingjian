//! 云联想 / 问字 / 翻译请求。

use super::*;
use crate::engine::prediction::split_local_scores;

#[test]
fn question_mode_asks_the_cloud_and_shows_answers_unvalidated() {
    let submitted = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let answer = CloudWord {
        text: "森".into(),
        syllables: Vec::new(),
        reading: Some("sēn".into()),
        local: false,
    };
    let restated = CloudWord {
        text: "木木木是什么字".into(),
        syllables: Vec::new(),
        reading: None,
        local: false,
    };
    let predictor = EchoPredictor {
        submitted: submitted.clone(),
        replies: vec![Prediction {
            sequence: 1,
            words: vec![restated, answer],
            sentence: None,
        }],
        sentence: true,
    };
    let mut engine = self::engine().with_predictor(Box::new(predictor));
    // 缺省 `?` 不是入口：开了开关才进问字
    assert!(!engine.takes_question_mark());
    engine.set_mode_keys(ModeKeys {
        question_mark: true,
        ..ModeKeys::default()
    });
    assert!(engine.takes_question_mark());
    engine.push('?');
    assert!(engine.bare_question());
    assert!(engine.question_mode());
    for c in "mumumu".chars() {
        engine.push(c);
    }
    assert!(!engine.bare_question());
    let query = engine.query().unwrap();
    assert!(query.candidates.items.is_empty());
    assert_eq!(query.marked_text(), "?mu'mu'mu");
    assert_eq!(query.marked_cursor(), 9);

    assert_eq!(engine.request_prediction(None, &[]), Some(1));
    let request = submitted.borrow()[0].clone();
    assert_eq!(request.kind, PredictionKind::Question);
    assert_eq!(request.pinyin, "mu'mu'mu");
    assert_eq!(request.letters, "mumumu");
    assert!(!request.want_sentence);

    // 答案的拼音与敲的字母对不上，但问字模式不校验；复述问题的「答案」剔掉
    let prediction = engine.poll_prediction().unwrap();
    assert_eq!(prediction.words.len(), 1);
    assert_eq!(prediction.words[0].text, "森");
    let answer = prediction
        .words
        .into_iter()
        .next()
        .unwrap()
        .into_candidate();
    assert_eq!(answer.reading.as_deref(), Some("sēn"));
    assert_eq!(engine.commit(&answer), "森");
    assert!(engine.composition().is_empty());
}

#[test]
fn prediction_request_trims_context_and_only_fires_while_composing() {
    let submitted = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let mut engine = engine().with_predictor(Box::new(EchoPredictor {
        submitted: submitted.clone(),
        replies: Vec::new(),
        sentence: true,
    }));
    engine.set_input("kaifa");
    let query = engine.query().unwrap();
    let surrounding = SurroundingText {
        before: "我们今天一起来".into(),
        after: "吧，好不好".into(),
    };
    let sequence = engine.request_prediction(Some(surrounding), &query.candidates.items);
    assert_eq!(sequence, Some(1));
    let request = submitted.borrow()[0].clone();
    assert_eq!(request.before, "天一起来");
    assert_eq!(request.after, "吧，");
    assert_eq!(request.pinyin, "kai'fa");
    assert_eq!(request.letters, "kaifa");
    assert_eq!(request.syllables, 2);
    assert_eq!(request.candidates[0], "开发");
    assert_eq!(request.max_items, 2);
    assert!(request.want_sentence);

    // 拼音太短不发
    engine.set_input("k");
    assert_eq!(engine.request_prediction(None, &[]), None);
    engine.set_input("kaifa");
    // 没有应用上下文也照发：词候选和整句都只靠拼音；本地历史不进请求
    let kaifa = query.candidates.items[0].clone();
    engine.commit(&kaifa);
    engine.punctuate('.');
    engine.set_input("kaifa");
    assert_eq!(engine.request_prediction(None, &[]), Some(3));
    let request = submitted.borrow()[1].clone();
    assert!(request.before.is_empty());
    assert!(request.want_sentence);
    // 上屏之后不联想
    engine.clear();
    assert_eq!(engine.request_prediction(None, &[]), None);
}

#[test]
fn cursor_in_the_middle_scopes_candidates_and_prediction_to_the_left_part() {
    let mut engine = engine();
    engine.set_input("kaifazhe");
    for _ in 0..3 {
        engine.move_cursor_left();
    }
    // kaifa|zhe：候选只看 kaifa（与单独打 kaifa 一样），zhe 只画出来
    let query = engine.query().unwrap();
    assert_eq!(query.candidates.items[0].text, "开发");
    assert_eq!(query.marked_text(), "kai'fa'zhe");
    assert_eq!(query.marked_cursor(), 6);

    // 上屏 开发 后剩 zhe，光标落到末尾，接着打就是往后加
    let kaifa = query.candidates.items[0].clone();
    assert_eq!(engine.commit(&kaifa), "开发");
    assert_eq!(engine.composition().text(), "zhe");
    assert_eq!(engine.composition().cursor(), 3);

    // 光标在开头时按整段算
    engine.set_input("kaifa");
    engine.move_cursor_home();
    assert_eq!(engine.query().unwrap().candidates.items[0].text, "开发");
}

#[test]
fn prediction_request_uses_the_scope_only() {
    let submitted = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let predictor = EchoPredictor {
        submitted: submitted.clone(),
        replies: Vec::new(),
        sentence: false,
    };
    let mut engine = engine().with_predictor(Box::new(predictor));
    engine.set_input("kaifazhe");
    for _ in 0..3 {
        engine.move_cursor_left();
    }
    assert_eq!(engine.request_prediction(None, &[]), Some(1));
    let request = submitted.borrow()[0].clone();
    assert_eq!(request.pinyin, "kai'fa");
    assert_eq!(request.letters, "kaifa");
    assert_eq!(request.syllables, 2);
}

#[test]
fn stale_predictions_are_dropped_and_accept_clears_composition() {
    let submitted = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let mut engine = engine().with_predictor(Box::new(EchoPredictor {
        submitted,
        sentence: false,
        replies: vec![
            Prediction {
                sequence: 2,
                words: vec![
                    cloud("开花", &["kai", "hua"]),
                    cloud("凯发", &["kai", "fa"]),
                ],
                sentence: Some("开发输入法".into()),
            },
            Prediction {
                sequence: 1,
                words: Vec::new(),
                sentence: Some("旧结果".into()),
            },
        ],
    }));
    engine.set_input("kaifa");
    engine.request_prediction(None, &[]);
    engine.request_prediction(None, &[]);
    let prediction = engine.poll_prediction().unwrap();
    assert_eq!(prediction.sequence, 2);
    assert_eq!(prediction.sentence.as_deref(), Some("开发输入法"));
    // 开花 的拼音对不上 kaifa，被过滤
    assert_eq!(prediction.words, [cloud("凯发", &["kai", "fa"])]);
    assert_eq!(engine.poll_prediction(), None);

    // 取消后连当前序号的结果也不要
    engine.request_prediction(None, &[]);
    engine.cancel_prediction();
    assert_eq!(engine.poll_prediction(), None);

    assert_eq!(engine.accept_prediction("开发输入法"), "开发输入法");
    assert!(engine.composition().is_empty());
    assert_eq!(engine.history().text(), "开发输入法");
}

#[test]
fn cloud_words_tolerate_typos_but_not_unrelated_words() {
    let submitted = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let mut engine = engine().with_predictor(Box::new(EchoPredictor {
        submitted,
        sentence: false,
        replies: vec![Prediction {
            sequence: 1,
            words: vec![
                cloud("这个东西吗", &["zhe", "ge", "dong", "xi", "ma"]),
                cloud("知道", &["zhi", "dao"]),
            ],
            sentence: None,
        }],
    }));
    engine.set_input("zhgdoima");
    engine.request_prediction(None, &[]);
    let prediction = engine.poll_prediction().unwrap();
    assert_eq!(
        prediction.words,
        [cloud("这个东西吗", &["zhe", "ge", "dong", "xi", "ma"])]
    );
    // 云端词上屏吃掉整段（按音节对不上的）拼音
    let word = Candidate {
        text: "这个东西吗".into(),
        kind: CandidateKind::Cloud,
        syllables: prediction.words[0].syllables.clone(),
        reading: None,
        translation: None,
    };
    assert_eq!(engine.commit(&word), "这个东西吗");
    assert!(engine.composition().is_empty());
}

#[test]
fn cloud_words_are_validated_against_abbreviated_pinyin() {
    let submitted = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let mut engine = engine().with_predictor(Box::new(EchoPredictor {
        submitted,
        sentence: false,
        replies: vec![Prediction {
            sequence: 1,
            words: vec![
                cloud("账套", &["zhang", "tao"]),
                cloud("张涛涛", &["zhang", "tao", "tao"]),
                cloud("知道", &["zhi", "dao"]),
                cloud("不是音节", &["zx", "tq", "a", "b"]),
            ],
            sentence: None,
        }],
    }));
    engine.set_input("zt");
    engine.request_prediction(None, &[]);
    let prediction = engine.poll_prediction().unwrap();
    assert_eq!(prediction.words, [cloud("账套", &["zhang", "tao"])]);
}

#[test]
fn accepted_sentence_completion_feeds_the_personal_ngram_and_can_be_retracted() {
    let shared = Arc::new(Mutex::new((Vec::new(), sentence::UserNgram::default())));
    let mut engine = Engine::new(Dictionary::parse(SAMPLE).unwrap())
        .with_language_model(Box::new(SentenceModel))
        .with_learner(Box::new(WordLearner {
            shared: shared.clone(),
            ..WordLearner::default()
        }));
    engine.set_input("kaifa");
    assert_eq!(
        engine.accept_prediction("开发输入法很好用。"),
        "开发输入法很好用。"
    );
    {
        let ngram = &shared.lock().unwrap().1;
        assert_eq!(ngram.pair(None, "开发"), 1);
        assert_eq!(ngram.pair(Some("开发"), "输入法"), 1);
        assert_eq!(ngram.pair(Some("输入法"), "很"), 1);
        assert_eq!(ngram.pair(Some("很"), "好用"), 1);
    }
    // 句尾是句号：下一个词按句首记
    assert_eq!(engine.chain.previous(), None);

    // 整句退格删光、同一段拼音重新选词：撤销这句记的转移
    for _ in 0.."开发输入法很好用。".chars().count() {
        engine.note_backspace();
    }
    engine.set_input("kaifa");
    let query = engine.query().unwrap();
    let first = query.candidates.items[0].clone();
    engine.commit(&first);
    let ngram = &shared.lock().unwrap().1;
    assert_eq!(ngram.pair(Some("开发"), "输入法"), 0);
    assert_eq!(ngram.pair(Some("很"), "好用"), 0);
}

#[test]
fn question_key_answers_code_points_locally_and_keeps_question_mark_alias() {
    let submitted = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let mut engine = engine().with_predictor(Box::new(EchoPredictor {
        submitted: submitted.clone(),
        replies: Vec::new(),
        sentence: false,
    }));
    engine.set_input("u4e00");
    assert!(engine.question_mode());
    assert!(engine.unicode_entry());
    let query = engine.query().unwrap();
    assert_eq!(query.candidates.items[0].text, "一");
    assert_eq!(query.candidates.items[0].kind, CandidateKind::Shortcut);
    assert_eq!(query.tail, "u4e00");
    // 码点本地就答，不问云端
    assert!(
        engine
            .request_prediction(None, &query.candidates.items)
            .is_none()
    );
    assert!(submitted.borrow().is_empty());

    engine.set_input("usangemu");
    assert!(engine.question_mode());
    assert!(!engine.unicode_entry());
    let query = engine.query().unwrap();
    assert!(query.candidates.items.is_empty());
    let body = query.tail.strip_prefix('u').unwrap().to_owned();
    assert!(!body.is_empty());

    // `?` 缺省不是入口：`?sangemu` 是英文直输段而不是问题
    engine.set_input("?sangemu");
    assert!(!engine.question_mode() && engine.raw_mode());
    // 开了开关才是别名：同一个问题、同样的切分，只是前缀不同
    engine.set_mode_keys(ModeKeys {
        question_mark: true,
        ..ModeKeys::default()
    });
    assert!(engine.question_mode());
    assert_eq!(engine.query().unwrap().tail, format!("?{body}"));

    // 换成 i 问字、v 表达式后 u 就是普通字母
    engine.set_mode_keys(ModeKeys {
        expression: 'v',
        question: 'i',
        question_mark: false,
    });
    engine.set_input("u4e00");
    assert!(!engine.question_mode());
    engine.set_input("i4e00");
    assert_eq!(engine.query().unwrap().candidates.items[0].text, "一");
    // 非法组合退回缺省
    engine.set_mode_keys(ModeKeys {
        expression: 'u',
        question: 'u',
        question_mark: false,
    });
    assert_eq!(engine.mode_keys(), ModeKeys::default());
}

#[test]
fn committing_a_cloud_word_learns_it_and_it_ranks_first_next_time() {
    let mut engine = engine().with_learner(Box::new(WordLearner::default()));
    engine.set_input("zt");
    assert!(engine.query().unwrap().candidates.items.is_empty());
    let word = Candidate {
        text: "账套".into(),
        kind: CandidateKind::Cloud,
        syllables: vec!["zhang".into(), "tao".into()],
        reading: None,
        translation: None,
    };
    assert_eq!(engine.commit(&word), "账套");
    assert!(engine.composition().is_empty());

    engine.set_input("zhangtao");
    let all = texts_of(&engine);
    assert_eq!(all[0], "账套");
    engine.set_input("zt");
    assert_eq!(texts_of(&engine)[0], "账套");

    // 词库里已有的云端词不重复记
    engine.set_input("kaifa");
    let mut kaifa = engine.query().unwrap().candidates.items[0].clone();
    kaifa.kind = CandidateKind::Cloud;
    engine.commit(&kaifa);
    engine.set_input("kaifa");
    assert_eq!(texts_of(&engine).iter().filter(|t| *t == "开发").count(), 1);
}

#[test]
fn traditional_mode_preserves_original_text_across_queries() {
    let mut engine = engine()
        .with_predictor(Box::new(EchoPredictor {
            submitted: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
            sentence: false,
            replies: vec![Prediction {
                sequence: 1,
                words: vec![cloud("凯发", &["kai", "fa"])],
                sentence: None,
            }],
        }))
        .with_learner(Box::new(WordLearner::default()));

    engine.set_traditional_mode(true);
    engine.set_input("kaifa");
    engine.request_prediction(None, &[]);
    let prediction = engine.poll_prediction().unwrap();
    let cloud_text = prediction.words[0].text.clone();

    engine.query().unwrap(); // 第二次 query() 不应清空云端词的映射

    let word = Candidate {
        text: cloud_text,
        kind: CandidateKind::Cloud,
        syllables: vec!["kai".into(), "fa".into()],
        reading: None,
        translation: None,
    };
    assert_eq!(engine.commit(&word), "凱發");
    // 检查词库里学到的是简体「凯发」
    assert!(engine.learner().weight("凯发") > 0);
    assert_eq!(engine.learner().weight("凱發"), 0);
}

#[test]
fn cloud_words_are_learned_with_the_typed_reading_when_it_fits() {
    let mut engine = engine().with_learner(Box::new(WordLearner::default()));
    let cloud_word = |text: &str, syllables: &[&str]| Candidate {
        text: text.into(),
        kind: CandidateKind::Cloud,
        syllables: syllables.iter().map(|s| (*s).to_owned()).collect(),
        reading: None,
        translation: None,
    };
    let has = |engine: &Engine, text: &str| texts_of(engine).iter().any(|t| t == text);
    // 模型把 先 的读音给成了 xia：敲的 kaixian 切得开、每个音节都是那个字的读音，按敲的学
    engine.set_input("kaixian");
    engine.commit(&cloud_word("开先", &["kai", "xia"]));
    engine.set_input("kaixian");
    assert!(has(&engine, "开先"));
    let user = engine.learner().user_words().unwrap();
    assert!(
        user.lookup(&["kai", "xian"], false)
            .iter()
            .any(|m| m.exact && m.text == "开先")
    );
    assert!(!user.lookup(&["kai", "xia"], false).iter().any(|m| m.exact));
    // 敲错了（kaixan 切不开）：模型的读音每个字都对得上，按模型的学
    engine.set_input("kaixan");
    engine.commit(&cloud_word("开想", &["kai", "xiang"]));
    engine.set_input("kaixiang");
    assert!(has(&engine, "开想"));
    // 敲错了、模型的读音又不是这个字的：不学
    engine.set_input("xiangxan");
    engine.commit(&cloud_word("想先", &["xiang", "xia"]));
    let user = engine.learner().user_words().unwrap();
    for reading in [["xiang", "xia"], ["xiang", "xian"]] {
        assert!(
            user.lookup(&reading, false)
                .iter()
                .all(|m| m.text != "想先")
        );
    }
}

#[test]
fn no_predictor_never_requests() {
    let mut engine = engine();
    assert!(!engine.prediction_enabled());
    engine.set_input("kaifa");
    assert_eq!(engine.request_prediction(None, &[]), None);
}

#[test]
fn translation_requests_carry_the_text_and_target_language() {
    use std::sync::{Arc, Mutex};
    struct Recorder(Arc<Mutex<Vec<PredictionRequest>>>);
    impl Predictor for Recorder {
        fn policy(&self) -> PredictionPolicy {
            PredictionPolicy::default()
        }
        fn submit(&mut self, request: PredictionRequest) {
            self.0.lock().unwrap().push(request);
        }
        fn poll(&mut self) -> Option<Prediction> {
            None
        }
    }
    let sent = Arc::new(Mutex::new(Vec::new()));
    let mut engine = Engine::new(Dictionary::parse(SAMPLE).unwrap())
        .with_predictor(Box::new(Recorder(sent.clone())));
    assert!(engine.request_translation("  ").is_none());
    let sequence = engine.request_translation("我想去吃饭").unwrap();
    let requests = sent.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].sequence, sequence);
    assert_eq!(requests[0].kind, PredictionKind::Translate);
    assert_eq!(requests[0].text, "我想去吃饭");
    assert_eq!(requests[0].target_language, "en");
    drop(requests);
    // 外文选区译回中文
    engine.request_translation("I want to eat.").unwrap();
    assert_eq!(sent.lock().unwrap()[1].target_language, "zh");
}

#[test]
fn bare_question_restores_punctuation_once_in_each_mode() {
    for (english, full_width, expected) in [
        (false, false, "?"),
        (false, true, "？"),
        (true, false, "?"),
        (true, true, "?"),
    ] {
        let mut engine = engine();
        engine.set_full_width_punctuation(full_width);
        engine.push('?');
        assert_eq!(
            engine.restore_bare_question(english).as_deref(),
            Some(expected)
        );
        assert!(engine.composition().is_empty());
        assert_eq!(engine.history().text(), expected);
        assert_eq!(engine.passthrough_pending, expected);
        assert_eq!(engine.restore_bare_question(english), None);
        assert_eq!(engine.history().text(), expected);
        assert_eq!(engine.passthrough_pending, expected);
    }
}

#[test]
fn restoring_question_preserves_other_compositions() {
    for input in ["", "nihao", "?nihao"] {
        let mut engine = engine();
        engine.set_input(input);
        assert_eq!(engine.restore_bare_question(false), None);
        assert_eq!(engine.composition().text(), input);
        assert!(engine.history().text().is_empty());
        assert!(engine.passthrough_pending.is_empty());
    }
}

/// 已知几个词、能给「开发者」出后继的小语言模型。
struct TinyLm;

impl LanguageModel for TinyLm {
    fn log_prob(&self, _previous: Option<&str>, word: &str) -> Option<f64> {
        matches!(word, "开发者" | "顺利" | "很" | "好").then_some(-1.0)
    }

    fn successors(&self, previous: &str, limit: usize) -> Vec<(String, u32)> {
        let all = if previous == "开发者" {
            vec![("顺利".to_owned(), 20), ("很".to_owned(), 15)]
        } else {
            Vec::new()
        };
        all.into_iter().take(limit).collect()
    }
}

/// 偏爱某个文本的打分器（rescoring 测试里的 Prefers 在另一个模块，这里单写一份）。
struct PrefersText(&'static str);

impl SentenceScorer for PrefersText {
    fn score(&self, _context: &str, texts: &[&str]) -> Vec<f64> {
        texts
            .iter()
            .map(|t| if *t == self.0 { -1.0 } else { -20.0 })
            .collect()
    }
}

#[test]
fn local_prediction_offers_a_sentence_completion_from_the_model() {
    let mut engine = engine()
        .with_language_model(Box::new(TinyLm))
        .with_async_sentence_scorer(Box::new(PrefersText("开发者顺利")), Some(0.5), None, None);
    engine.set_local_prediction(true);
    engine.set_input("kaifazhe");
    engine.query().unwrap();
    // 云联想没开、本地开着：照发
    let sequence = engine.request_prediction(None, &[]).unwrap();
    let started = std::time::Instant::now();
    let prediction = loop {
        if let Some(prediction) = engine.poll_prediction() {
            break prediction;
        }
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "本地联想没回结果"
        );
        std::thread::sleep(Duration::from_millis(5));
    };
    assert_eq!(prediction.sequence, sequence);
    // 最好的续写胜过基线：整句补全 = guess + 续写；本地不出词槽
    assert!(prediction.words.is_empty());
    assert_eq!(prediction.sentence.as_deref(), Some("开发者顺利"));
}

#[test]
fn short_inputs_and_missing_model_do_not_ask_for_local_predictions() {
    let mut engine = engine().with_language_model(Box::new(TinyLm));
    engine.set_local_prediction(true);
    // 两个音节：短词门控，不唤醒模型，云也没开就不发
    engine.set_input("kaifa");
    engine.query().unwrap();
    assert_eq!(engine.request_prediction(None, &[]), None);
    // 三个音节但没有后台模型：也不发
    engine.set_input("kaifazhe");
    engine.query().unwrap();
    assert_eq!(engine.request_prediction(None, &[]), None);
}

#[test]
fn local_prediction_is_off_by_default_and_cancel_drops_the_in_flight_request() {
    let mut engine = engine()
        .with_language_model(Box::new(TinyLm))
        .with_async_sentence_scorer(Box::new(PrefersText("开发者顺利")), Some(0.5), None, None);
    engine.set_input("kaifazhe");
    engine.query().unwrap();
    // 缺省关：不发
    assert_eq!(engine.request_prediction(None, &[]), None);
    engine.set_local_prediction(true);
    assert!(engine.request_prediction(None, &[]).is_some());
    // 作废之后序号对不上，结果不会再被收：轮询一小段时间，什么都没等到就是对的
    engine.cancel_prediction();
    let started = std::time::Instant::now();
    while started.elapsed() < Duration::from_millis(300) {
        assert!(
            engine.poll_prediction().is_none(),
            "作废后的本地联想结果不该被收下"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn split_local_scores_demotes_continuations_and_surfaces_deep_words() {
    // 分段:基线 + 两个接续 + 两个参照 + 三个深池词
    let proposals = vec!["顺利".to_owned(), "很".to_owned()];
    let refs = vec!["开发者".to_owned(), "开发".to_owned()];
    let words = vec![
        (
            "目标词甲".to_owned(),
            vec!["kai".to_owned(), "fa".to_owned(), "zhe".to_owned()],
        ),
        (
            "目标词乙".to_owned(),
            vec!["kai".to_owned(), "fa".to_owned(), "zhe".to_owned()],
        ),
        (
            "目标词丙".to_owned(),
            vec!["kai".to_owned(), "fa".to_owned(), "zhe".to_owned()],
        ),
    ];
    // 基线 -5;接续一个胜基线(-3)、一个不胜(-6);参照都是 -2;深池 -1 / -1.5 / -3
    let scores = vec![-5.0, -3.0, -6.0, -2.0, -2.0, -1.0, -1.5, -3.0];
    let (sentence, words) =
        split_local_scores(&scores, &proposals, &refs, &words, &[], 2, "开发者");
    // 接续:只有胜基线的那个出,拼在 guess 后
    assert_eq!(sentence.as_deref(), Some("开发者顺利"));
    // 深池:胜过参照(-2)的只有甲乙;按分排序、slots=2 截断
    assert_eq!(words.len(), 2);
    assert_eq!(words[0].text, "目标词甲");
    assert_eq!(words[1].text, "目标词乙");
    assert_eq!(
        words[0].syllables,
        vec!["kai".to_owned(), "fa".to_owned(), "zhe".to_owned()]
    );

    // 深池全都压不过参照:一个都不捞
    let deep = vec![
        (
            "目标词甲".to_owned(),
            vec!["kai".to_owned(), "fa".to_owned(), "zhe".to_owned()],
        ),
        (
            "目标词乙".to_owned(),
            vec!["kai".to_owned(), "fa".to_owned(), "zhe".to_owned()],
        ),
        (
            "目标词丙".to_owned(),
            vec!["kai".to_owned(), "fa".to_owned(), "zhe".to_owned()],
        ),
    ];
    let scores = vec![-5.0, -3.0, -6.0, -1.0, -1.0, -2.0, -2.0, -2.5];
    let (_, words) = split_local_scores(&scores, &proposals, &refs, &deep, &[], 2, "开发者");
    assert!(words.is_empty());

    // slots = 0:不要词槽
    let scores = vec![-5.0, -3.0, -6.0, -2.0, -2.0, -1.0, -1.5, -3.0];
    let (_, words) = split_local_scores(&scores, &proposals, &refs, &deep, &[], 0, "开发者");
    assert!(words.is_empty());

    // 没有接续提议时照常出词槽
    let scores = vec![-5.0, -2.0, -2.0, -1.0];
    let (sentence, words) = split_local_scores(
        &scores,
        &[],
        &refs,
        &[("目标词甲".to_owned(), vec!["kai".to_owned()])],
        &[],
        2,
        "开发者",
    );
    assert!(sentence.is_none());
    assert_eq!(words.len(), 1);

    // 纠错变体：要高出参照 CORRECTION_MARGIN(2.0) 才出——参照 -2，阈值 0
    // 纠错变体与 guess 等长，跟基线比：高出 CORRECTION_MARGIN(2.0) 才出——基线 -5，阈值 -3
    let corrections = vec![
        ("纠错甲".to_owned(), vec!["kai".to_owned()]),
        ("纠错乙".to_owned(), vec!["kai".to_owned()]),
    ];
    let scores = vec![-5.0, -2.0, -2.0, 1.0, -4.0];
    let (sentence, words) = split_local_scores(&scores, &[], &refs, &[], &corrections, 2, "开发者");
    assert!(sentence.is_none());
    assert_eq!(words.len(), 1);
    assert_eq!(words[0].text, "纠错甲");

    // 深池词按「每字平均分」跟参照比（长候选不吃亏），再与纠错变体同池按分排序
    let deep = vec![("深池甲".to_owned(), vec!["kai".to_owned()])];
    let scores = vec![-5.0, -2.0, -2.0, -1.0, 1.0, -4.0];
    let (_, words) = split_local_scores(&scores, &[], &refs, &deep, &corrections, 2, "开发者");
    assert_eq!(words.len(), 2);
    assert_eq!(words[0].text, "纠错甲");
    assert_eq!(words[1].text, "深池甲");

    // 同样两字候选：长参照按每字平均不再压倒短候选（旧的累加分会让 3 字候选永远输）
    let long_refs = vec!["一二三四五六".to_owned()];
    let short_deep = vec![("短词".to_owned(), vec!["kai".to_owned()])];
    // 参照 6 字 -6（每字 -1）；短词 2 字 -1.5（每字 -0.75）→ 胜出
    let scores = vec![-5.0, -6.0, -1.5];
    let (_, words) = split_local_scores(&scores, &[], &long_refs, &short_deep, &[], 2, "开发者");
    assert_eq!(words.len(), 1);
    assert_eq!(words[0].text, "短词");
}

/// 只知道几个单字词的小模型：让「开发者」切成「开发 + 者」两个词。
struct HomoLm;

impl LanguageModel for HomoLm {
    fn log_prob(&self, _previous: Option<&str>, word: &str) -> Option<f64> {
        matches!(word, "开发" | "者" | "着" | "这").then_some(-1.0)
    }
}

#[test]
fn local_prediction_corrects_a_word_with_a_same_reading_alternative() {
    // 「者」与「着 / 这」同音：最优路径切出的词逐位换同音词，应出纠错候选
    const HOMO: &str = "开发	kai fa	9000\n者	zhe	8000\n着	zhe	5000\n这	zhe	6000\n";
    let mut engine = Engine::new(Dictionary::parse(HOMO).unwrap())
        .with_language_model(Box::new(HomoLm))
        .with_async_sentence_scorer(Box::new(PrefersText("开发着")), Some(0.5), None, None);
    engine.set_local_prediction(true);
    engine.set_input("kaifazhe");
    let query = engine.query().unwrap();
    let sequence = engine
        .request_prediction(None, &query.candidates.items)
        .unwrap();
    let started = std::time::Instant::now();
    let prediction = loop {
        if let Some(prediction) = engine.poll_prediction() {
            break prediction;
        }
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "本地联想没回结果"
        );
        std::thread::sleep(Duration::from_millis(5));
    };
    assert_eq!(prediction.sequence, sequence);
    // 模型偏爱「开发着」：它高出参照（第一页候选）余量，纠错候选进词槽
    assert_eq!(prediction.sentence, None);
    assert_eq!(prediction.words.len(), 1);
    assert_eq!(prediction.words[0].text, "开发着");
    assert_eq!(
        prediction.words[0].syllables,
        vec!["kai".to_owned(), "fa".to_owned(), "zhe".to_owned()]
    );
    // 本机模型给的：候选标 Local，界面上不画云朵（别让人以为数据外传）
    assert!(prediction.words[0].local);
    let candidate = prediction.words[0].clone().into_candidate();
    assert_eq!(candidate.kind, CandidateKind::Local);
}

#[test]
fn local_prediction_surfaces_deep_pool_words_beating_the_page() {
    // 11 个同读音词:目标词频率最低,排在第一页(9 条)之外,只能靠深池捞
    const DEEP: &str = "开发者\tkai fa zhe\t3000\n\
        开着发\tkai fa zhe\t500\n\
        开发着\tkai fa zhe\t450\n\
        开法者\tkai fa zhe\t400\n\
        者开发\tkai fa zhe\t350\n\
        发开者\tkai fa zhe\t300\n\
        开者发\tkai fa zhe\t250\n\
        发者开\tkai fa zhe\t200\n\
        者发开\tkai fa zhe\t150\n\
        开者子\tkai fa zhe\t100\n\
        开发丙\tkai fa zhe\t20\n";
    let mut engine = Engine::new(Dictionary::parse(DEEP).unwrap())
        .with_language_model(Box::new(TinyLm))
        .with_async_sentence_scorer(Box::new(PrefersText("开发丙")), Some(0.5), None, None);
    engine.set_local_prediction(true);
    engine.set_input("kaifazhe");
    let query = engine.query().unwrap();
    let sequence = engine
        .request_prediction(None, &query.candidates.items)
        .unwrap();
    let started = std::time::Instant::now();
    let prediction = loop {
        if let Some(prediction) = engine.poll_prediction() {
            break prediction;
        }
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "本地联想没回结果"
        );
        std::thread::sleep(Duration::from_millis(5));
    };
    assert_eq!(prediction.sequence, sequence);
    // 深池词胜过第一页参照:捞进词槽;接续全部不胜基线:没有整句
    assert_eq!(prediction.sentence, None);
    assert_eq!(prediction.words.len(), 1);
    assert_eq!(prediction.words[0].text, "开发丙");
    assert_eq!(
        prediction.words[0].syllables,
        vec!["kai".to_owned(), "fa".to_owned(), "zhe".to_owned()]
    );

    // 参照本身就是模型偏爱的:深池全被压住,词槽为空
    let mut engine = Engine::new(Dictionary::parse(DEEP).unwrap())
        .with_language_model(Box::new(TinyLm))
        .with_async_sentence_scorer(Box::new(PrefersText("开发者")), Some(0.5), None, None);
    engine.set_local_prediction(true);
    engine.set_input("kaifazhe");
    let query = engine.query().unwrap();
    let sequence = engine
        .request_prediction(None, &query.candidates.items)
        .unwrap();
    let started = std::time::Instant::now();
    let prediction = loop {
        if let Some(prediction) = engine.poll_prediction() {
            break prediction;
        }
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "本地联想没回结果"
        );
        std::thread::sleep(Duration::from_millis(5));
    };
    assert_eq!(prediction.sequence, sequence);
    assert!(prediction.words.is_empty());
    assert_eq!(prediction.sentence, None);
}
