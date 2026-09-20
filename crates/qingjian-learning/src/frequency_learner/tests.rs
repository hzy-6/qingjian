use super::*;
use qingjian_core::Engine;
use qingjian_dictionary::Dictionary;

fn candidate(text: &str) -> Candidate {
    Candidate {
        text: text.to_owned(),
        kind: qingjian_core::CandidateKind::Chinese,
        syllables: Vec::new(),
        reading: None,
        translation: None,
    }
}

#[test]
fn records_and_round_trips_through_tsv() {
    let mut learner = FrequencyLearner::default();
    learner.record(&candidate("开发"));
    learner.record(&candidate("开发"));
    learner.record(&candidate("中文"));
    assert_eq!(learner.weight("开发"), 2);
    assert!(learner.is_dirty());

    let dir = std::env::temp_dir().join(format!("qingjian-learning-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("user.tsv");
    learner.save_to(&path).unwrap();
    assert!(!learner.is_dirty());

    let mut restored = FrequencyLearner::from_path(&path).unwrap();
    assert_eq!(restored.weight("开发"), 2);
    assert_eq!(restored.weight("中文"), 1);
    assert_eq!(restored.weight("没有"), 0);

    // flush 写回加载时的路径
    restored.record(&candidate("中文"));
    restored.flush();
    assert_eq!(
        FrequencyLearner::from_path(&path).unwrap().weight("中文"),
        2
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn choices_are_keyed_by_input_and_round_trip() {
    let dir = std::env::temp_dir().join("qingjian-user-choices-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("user.tsv");
    let _ = std::fs::remove_file(dir.join(USER_CHOICES_FILE));

    let mut learner = FrequencyLearner::from_path(&path).unwrap();
    learner.record_choice("ba", "吧");
    learner.record_choice("ba", "吧");
    learner.record_choice("bazhege", "把");
    learner.record_raw("nihooma");
    assert_eq!(learner.raw_count("nihooma"), 1);
    assert_eq!(learner.choice_weight("ba", "吧"), 2);
    assert_eq!(learner.choice_weight("ba", "把"), 0);
    assert_eq!(learner.choice_weight("bazhege", "把"), 1);
    learner.flush();

    let reloaded = FrequencyLearner::from_path(&path).unwrap();
    assert_eq!(reloaded.choice_count(), 3);
    assert_eq!(reloaded.choice_weight("ba", "吧"), 2);
    assert_eq!(reloaded.raw_count("nihooma"), 1);
    assert_eq!(reloaded.choice_weight("bazhege", "把"), 1);
}

#[test]
fn unrecord_reverses_each_kind_of_record() {
    let mut learner = FrequencyLearner::default();
    let candidate = Candidate {
        text: "开放".into(),
        kind: qingjian_core::CandidateKind::Chinese,
        syllables: vec!["kai".into(), "fang".into()],
        reading: None,
        translation: None,
    };
    learner.record(&candidate);
    learner.record_choice("kaifa", "开放");
    learner.record_transition(Context::START, "开放", 2);
    learner.unrecord("开放");
    learner.unrecord_choice("kaifa", "开放");
    learner.unrecord_transition(Context::START, "开放", 2);
    assert_eq!(learner.weight("开放"), 0);
    assert_eq!(learner.choice_weight("kaifa", "开放"), 0);
    assert_eq!(learner.choice_count(), 0);
    assert!(learner.user_ngram().is_none());
    // 没记过的撤销不会变成负数
    learner.unrecord("没有");
    learner.unrecord_choice("x", "没有");
    assert_eq!(learner.weight("没有"), 0);
}

#[test]
fn choices_decay_when_over_the_cap() {
    let mut learner = FrequencyLearner::default();
    learner.record_choice("a", "甲");
    learner.record_choice("a", "甲");
    learner.record_choice("a", "甲");
    learner.record_choice("a", "乙");
    learner.decay_choices();
    assert_eq!(learner.choice_weight("a", "甲"), 1);
    assert_eq!(learner.choice_weight("a", "乙"), 0);
    assert_eq!(learner.choice_count(), 1);
}

#[test]
fn transitions_round_trip_through_ngram_tsv() {
    let dir = std::env::temp_dir().join("qingjian-user-ngram-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("user.tsv");
    let _ = std::fs::remove_file(dir.join(USER_NGRAM_FILE));

    let mut learner = FrequencyLearner::from_path(&path).unwrap();
    assert!(learner.user_ngram().is_none());
    learner.record_transition(Context::START, "我", 1);
    learner.record_transition(Context::after("我"), "想", 1);
    learner.record_transition(Context::after("我"), "想", 1);
    learner.record_transition(Context::after_two("我", "想"), "去", 1);
    assert_eq!(learner.user_ngram().unwrap().pair(Some("我"), "想"), 2);
    learner.flush();

    let reloaded = FrequencyLearner::from_path(&path).unwrap();
    // 二元 <s>我 / 我想 / 想去，三元 <s>我想 / 我想去
    assert_eq!(reloaded.ngram_transition_count(), 5);
    assert_eq!(reloaded.user_ngram().unwrap().pair(None, "我"), 1);
    assert_eq!(reloaded.user_ngram().unwrap().count("想"), 2);
    assert_eq!(
        reloaded
            .user_ngram()
            .unwrap()
            .triple(Some("我"), "想", "去"),
        1
    );
}

#[test]
fn typos_round_trip_and_unrecord() {
    let dir = std::env::temp_dir().join("qingjian-user-typos-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("user.tsv");
    let _ = std::fs::remove_file(dir.join(USER_TYPOS_FILE));

    let mut learner = FrequencyLearner::from_path(&path).unwrap();
    learner.record_typo("gan", "guan");
    learner.record_typo("gan", "guan");
    learner.record_typo("shou", "shuo");
    learner.unrecord_typo("shou", "shuo");
    learner.unrecord_typo("mei", "you");
    assert_eq!(learner.typo_count("gan", "guan"), 2);
    assert_eq!(learner.typo_count("shou", "shuo"), 0);
    assert_eq!(learner.typo_count_total(), 1);
    learner.flush();

    let reloaded = FrequencyLearner::from_path(&path).unwrap();
    assert_eq!(reloaded.typo_count("gan", "guan"), 2);
    assert_eq!(reloaded.typo_count_total(), 1);
}

#[test]
fn broken_lines_are_skipped_instead_of_failing_the_load() {
    let dir = std::env::temp_dir().join("qingjian-broken-lines-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("user.tsv");
    std::fs::write(&path, "开发\t3\n没有次数\n中文\tabc\n中文\t1\n").unwrap();
    std::fs::write(dir.join(USER_NGRAM_FILE), "<s>\t我\t3\n坏行\n我\t想\t2\n").unwrap();
    std::fs::write(dir.join(USER_CHOICES_FILE), "ba\t吧\t2\nba\n").unwrap();
    std::fs::write(dir.join(USER_TYPOS_FILE), "gan\tguan\tx\ngan\tguan\t1\n").unwrap();
    std::fs::write(dir.join(USER_WORDS_FILE), "账套\tzhang tao\t100\n只有词\n").unwrap();
    // 编码坏掉的字节也不能让整个文件读不了
    std::fs::write(dir.join(USER_ENGLISH_FILE), b"gist\t2\n\xff\xfe\t1\n").unwrap();

    let learner = FrequencyLearner::from_path(&path).unwrap();
    assert_eq!(learner.weight("开发"), 3);
    assert_eq!(learner.weight("中文"), 1);
    assert_eq!(learner.user_ngram().unwrap().pair(Some("我"), "想"), 2);
    assert_eq!(learner.choice_weight("ba", "吧"), 2);
    assert_eq!(learner.typo_count("gan", "guan"), 1);
    assert_eq!(learner.word_count(), 1);
    assert_eq!(learner.english_count(), 2);
    assert!(learner.path.is_some());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn saving_is_atomic_and_leaves_no_temporary_files() {
    let dir = std::env::temp_dir().join("qingjian-atomic-save-test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("user.tsv");
    let mut learner = FrequencyLearner::from_path(&path).unwrap();
    learner.record(&candidate("开发"));
    learner.record_choice("kaifa", "开发");
    learner.record_transition(Context::START, "开发", 1);
    learner.record_typo("gan", "guan");
    learner.learn_word("账套", &["zhang".into(), "tao".into()]);
    learner.learn_english("gist");
    assert!(learner.has_unsaved());
    learner.flush();
    assert!(!learner.has_unsaved());
    let names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names.len(), 6, "{names:?}");
    assert!(names.iter().all(|name| !name.contains(".tmp")), "{names:?}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn missing_file_is_empty_table() {
    let learner = FrequencyLearner::from_path("/nonexistent/qingjian-user.tsv").unwrap();
    assert!(learner.is_empty());
}

#[test]
fn forget_removes_the_user_word_and_every_trace_of_learning() {
    let mut learner = FrequencyLearner::default();
    learner.learn_word("账套", &["zhang".into(), "tao".into()]);
    learner.record(&candidate("账套"));
    learner.record_choice("zhangtao", "账套");
    learner.record_choice("zt", "账套");
    learner.record_choice("zt", "周天");
    learner.record_transition(Context::START, "账套", 1);
    learner.record_transition(Context::after("账套"), "建好", 1);
    let forgotten = learner.forget("账套");
    assert!(forgotten.user_word && forgotten.learning);
    assert!(learner.user_words().is_none());
    assert_eq!(learner.weight("账套"), 0);
    assert_eq!(learner.choice_weight("zt", "账套"), 0);
    assert_eq!(learner.choice_weight("zt", "周天"), 1);
    assert!(learner.user_ngram().is_none());
    // 词库词、没学过：什么都没清
    assert!(learner.forget("开发").is_nothing());
    // 只学过、不是用户词
    learner.record(&candidate("开发"));
    let forgotten = learner.forget("开发");
    assert!(!forgotten.user_word && forgotten.learning);
    // 个人英文词
    learner.learn_english("gist");
    assert!(learner.forget_english("Gist"));
    assert!(!learner.forget_english("gist"));
    assert!(learner.user_english().is_none());
}

#[test]
fn user_words_round_trip_through_tsv_and_form_a_dictionary() {
    let dir = std::env::temp_dir().join("qingjian-user-words-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("user.tsv");
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(dir.join(USER_WORDS_FILE));

    let mut learner = FrequencyLearner::from_path(&path).unwrap();
    assert!(learner.user_words().is_none());
    learner.learn_word("账套", &["zhang".to_owned(), "tao".to_owned()]);
    learner.learn_word("账套", &["zhang".to_owned(), "tao".to_owned()]);
    assert_eq!(learner.word_count(), 1);
    let hits = learner
        .user_words()
        .unwrap()
        .lookup(&["zhang", "tao"], false);
    assert_eq!(hits[0].text, "账套");
    learner.flush();

    let reloaded = FrequencyLearner::from_path(&path).unwrap();
    assert_eq!(reloaded.word_count(), 1);
    let hits = reloaded.user_words().unwrap().lookup(&["zhang"], true);
    assert_eq!(hits[0].text, "账套");
}

#[test]
fn english_words_round_trip_through_tsv_and_form_a_word_list() {
    let dir = std::env::temp_dir().join("qingjian-user-english-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("user.tsv");
    let _ = std::fs::remove_file(FrequencyLearner::english_path(&path));
    let mut learner = FrequencyLearner::from_path(&path).unwrap();
    assert!(learner.user_english().is_none());
    learner.learn_english("gist");
    learner.learn_english("Gist");
    learner.learn_english("python");
    assert_eq!(learner.english_count(), 2);
    // 第一次敲的写法留下，次数合并
    assert_eq!(learner.user_english().unwrap().get("gist"), Some("gist"));
    assert_eq!(learner.user_english().unwrap().complete("g", 3), ["gist"]);
    learner.flush();
    let reloaded = FrequencyLearner::from_path(&path).unwrap();
    assert_eq!(reloaded.english_count(), 2);
    assert_eq!(
        reloaded.user_english().unwrap().get("python"),
        Some("python")
    );
    let saved = std::fs::read_to_string(FrequencyLearner::english_path(&path)).unwrap();
    assert!(saved.contains("gist\t2"));
}

/// 清空学习数据后排序恢复（产品语义：学习数据是数据目录里的文件，删掉即清空）。
/// 用带 Engine 的完整流程：连选 开发 三次并落盘 → 重开的引擎读到学习文件、开发 第一；
/// 删掉全部学习文件再重建 → 排序回到未学习状态（词频高的 开放 第一）。
#[test]
fn deleting_the_learning_files_restores_the_unlearned_ranking() {
    let dir = std::env::temp_dir().join("qingjian-clear-learning-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("user.tsv");
    let all_files = [
        path.clone(),
        dir.join(USER_WORDS_FILE),
        dir.join(USER_NGRAM_FILE),
        dir.join(USER_CHOICES_FILE),
        dir.join(USER_ENGLISH_FILE),
        dir.join(USER_TYPOS_FILE),
    ];
    for file in &all_files {
        let _ = std::fs::remove_file(file);
    }
    let dictionary = "开发\tkai fa\t9000\n开放\tkai fang\t20000\n";
    let build = || {
        Engine::new(Dictionary::parse(dictionary).unwrap())
            .with_learner(Box::new(FrequencyLearner::from_path(&path).unwrap()))
    };

    // 未学习：`kaif` 首选是词频更高的 开放
    let mut engine = build();
    engine.set_input("kaif");
    assert_eq!(engine.query().unwrap().candidates.items[0].text, "开放");

    // 连选三次 开发，学习落盘
    for _ in 0..3 {
        engine.set_input("kaif");
        let kaifa = engine
            .query()
            .unwrap()
            .candidates
            .items
            .iter()
            .find(|c| c.text == "开发")
            .cloned()
            .unwrap();
        engine.commit(&kaifa);
    }
    assert_eq!(engine.learner().weight("开发"), 3);
    engine.learner_mut().flush();

    // 重开的引擎读到学习文件：开发 升到第一
    let mut engine = build();
    engine.set_input("kaif");
    assert_eq!(engine.query().unwrap().candidates.items[0].text, "开发");
    assert_eq!(engine.learner().choice_weight("kaif", "开发"), 3);

    // 清空学习数据：删掉数据目录里的全部学习文件（没写出来的本来就不存在）后重建，排序回到未学习状态
    assert!(path.is_file(), "词频文件该已落盘");
    assert!(dir.join(USER_CHOICES_FILE).is_file(), "选择文件该已落盘");
    for file in &all_files {
        let _ = std::fs::remove_file(file);
    }
    let mut engine = build();
    engine.set_input("kaif");
    assert_eq!(engine.query().unwrap().candidates.items[0].text, "开放");
    assert_eq!(engine.learner().weight("开发"), 0);
    assert_eq!(engine.learner().choice_weight("kaif", "开发"), 0);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// 时间衰减:带日期的行按 90 天半衰期折算(90 天前剩一半、180 天前剩四分之一),
/// 不带日期的旧行照旧;写出的行带日期,本会话动过的词刷新为今天。
#[test]
fn counts_decay_by_last_seen_date() {
    let mut learner = FrequencyLearner::default();
    let old = format!("老词\t10\t{}", days_ago(120));
    let ancient = format!("古词\t8\t{}", days_ago(300));
    learner.load_counts(&format!("新词\t6\n{old}\n{ancient}\n"));
    assert_eq!(learner.weight("新词"), 6, "没有日期的行照旧");
    let d120 = days_ago(120);
    eprintln!(
        "DBG date={d120} days_since={:?}",
        crate::frequency_learner::tables::days_since(&d120)
    );
    eprintln!("DBG weight={}", learner.weight("老词"));
    assert_eq!(
        learner.weight("老词"),
        4,
        "120 天 ≈ 1.33 个半衰期,10 → 5 折成 3(向上取整后见半衰公式)"
    );
    assert!(learner.weight("古词") <= 1, "300 天前几乎衰减光");
}

/// 算「N 天前」的日期字符串(本地时区)。
fn days_ago(days: i64) -> String {
    use jiff::Zoned;
    use jiff::civil::Date;
    let today: Date = Zoned::now().date();
    let date = today
        .checked_add(jiff::Span::new().days(-days))
        .expect("日期减法");
    date.strftime("%Y-%m-%d").to_string()
}
