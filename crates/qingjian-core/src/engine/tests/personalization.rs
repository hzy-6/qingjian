//! 个性化学习对排序的影响：选择次数怎么抬升常选的词、什么不该被它压过、清空与重排的边界。

use super::*;

/// 连续几次选同一个词，它升到首位：`kaif` 未学习时词频高的 开放 在前，选过 开发 三次后 开发 第一。
#[test]
fn committing_a_word_repeatedly_lifts_it_to_the_top() {
    let mut engine = engine().with_learner(Box::new(CountingLearner(HashMap::new())));
    engine.set_input("kaif");
    assert_eq!(texts_of(&engine)[0], "开放");
    for round in 1..=3 {
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
        assert_eq!(engine.learner().weight("开发"), round);
    }
    engine.set_input("kaif");
    let all = texts_of(&engine);
    assert_eq!(all[0], "开发");
    assert_eq!(all[1], "开放");
}

/// 个人选择只影响词级排序键，压不过结构上更合适的候选：把 开发 在 `kaif` 下选了 10 次之后，
/// 更长的整句拼音仍出整句候选第 1，全拼覆盖更优的 开发者 仍排在只覆盖前缀的 开发 前面。
#[test]
fn a_heavily_chosen_short_word_stays_below_the_fuller_candidates() {
    let mut engine = engine().with_learner(Box::new(CountingLearner(HashMap::new())));
    for _ in 0..10 {
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
    assert_eq!(engine.learner().choice_weight("kaif", "开发"), 10);
    assert_eq!(engine.learner().weight("开发"), 10);

    // 更长的整句拼音：整句候选仍是第 1（整句插在词级候选之前，选择次数够不着它）
    engine.set_input("xiangkaifa");
    let query = engine.query().unwrap();
    assert_eq!(query.candidates.items[0].kind, CandidateKind::Sentence);
    assert_eq!(query.candidates.items[0].text, "想开发");

    // 全拼覆盖更优的词也不被越过的：kaifazhe 下 开发者（覆盖全段）仍在 开发（前缀）之前
    engine.set_input("kaifazhe");
    let all = texts_of(&engine);
    let developer = all.iter().position(|t| t == "开发者").unwrap();
    let kaifa = all.iter().position(|t| t == "开发").unwrap();
    assert!(developer < kaifa, "{all:?}");
}

/// 偏爱某个文本的假打分器，其余文本给低分。
struct ScorerPreferring(&'static str);

impl SentenceScorer for ScorerPreferring {
    fn score(&self, _context: &str, texts: &[&str]) -> Vec<f64> {
        texts
            .iter()
            .map(|t| if *t == self.0 { -1.0 } else { -20.0 })
            .collect()
    }
}

/// 模型重排不重复记学习：query → request_rescoring → poll_rescoring → 重排后再 query、补画译文，
/// 期间一条学习都不记；之后 commit 一次，record 与 choice 恰好各记 1 次。
#[test]
fn rescoring_queries_do_not_duplicate_learning() {
    let dictionary = Dictionary::parse(
        "开\tkai\t20000\n发\tfa\t3000\n开发\tkai fa\t9000\n先\txian\t10000\n现\txian\t4000\n",
    )
    .unwrap();
    let mut engine = Engine::new(dictionary)
        .with_learner(Box::new(CountingLearner(HashMap::new())))
        .with_async_sentence_scorer(Box::new(ScorerPreferring("开发先")), Some(0.5), None, None);
    // 第一查：整句路径（开发先 / 开发现）还没分，等着壳送后台
    engine.set_input("kaifaxian");
    let _ = engine.query().unwrap();
    assert!(engine.rescoring_pending());
    assert!(engine.request_rescoring());
    let started = std::time::Instant::now();
    while !engine.poll_rescoring() {
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "后台没回结果"
        );
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    // 重排后的分到了缓存：再查一次拿到重排后的整句，并像壳一样补画译文；全程不该记任何学习
    engine.set_input("kaifaxian");
    let mut query = engine.query().unwrap();
    engine.annotate(&mut query.candidates);
    assert_eq!(query.candidates.items[0].kind, CandidateKind::Sentence);
    assert_eq!(query.candidates.items[0].text, "开发先");
    assert_eq!(
        engine.learner().weight("开发先"),
        0,
        "query 与 annotate 不该记学习"
    );
    assert_eq!(engine.learner().weight("开发"), 0);
    assert_eq!(
        engine.learner().choice_weight("kaifa", "开发"),
        0,
        "query 与 annotate 不该记选择"
    );
    // 只 commit 一次：record 与 choice 各 1，别的词一条都没记
    let kaifa = query
        .candidates
        .items
        .iter()
        .find(|c| c.text == "开发" && c.kind == CandidateKind::Chinese)
        .cloned()
        .unwrap();
    assert_eq!(engine.commit(&kaifa), "开发");
    assert_eq!(engine.learner().weight("开发"), 1);
    assert_eq!(engine.learner().choice_weight("kaifa", "开发"), 1);
    assert_eq!(engine.learner().weight("开"), 0);
    assert_eq!(engine.learner().weight("先"), 0);
}

/// 选择按输入串与候选全拼各记一份:`kaif` 下选的 开发,`kaifa` 查询直接受益(反向同理)。
#[test]
fn choices_generalize_across_the_candidate_full_pinyin() {
    let mut engine = engine().with_learner(Box::new(CountingLearner(HashMap::new())));
    // 敲前缀 kaif 选 开发(词级候选):input=kaif,候选全拼 kaifa ≠ kaif → 双键
    engine.set_input("kaif");
    let query = engine.query().unwrap();
    let kaifa = query
        .candidates
        .items
        .iter()
        .find(|c| c.text == "开发")
        .cloned()
        .unwrap();
    engine.commit(&kaifa);
    assert_eq!(engine.learner().choice_weight("kaif", "开发"), 1);
    // 规范全拼键 kaifa 下也有一份
    assert_eq!(engine.learner().choice_weight("kaifa", "开发"), 1);
    // 全拼 kaifa 查询:开发 按自身全拼查到记录,排到 开放 前面(开放 的全拼是 kai fang,不共享这份)
    engine.set_input("kaifa");
    assert_eq!(texts_of(&engine)[0], "开发");
}

/// 负反馈:上屏后删掉换选别的词,被换掉的词在这个输入串下记负分——
/// 净数(正减负)变低,再选时它不再稳居第一。
#[test]
fn retracting_a_choice_demotes_the_replaced_word() {
    let mut engine = engine().with_learner(Box::new(CountingLearner(HashMap::new())));
    let pick = |engine: &mut Engine, input: &str, text: &str| {
        engine.set_input(input);
        let candidate = engine
            .query()
            .unwrap()
            .candidates
            .items
            .into_iter()
            .find(|c| c.text == text && c.kind == CandidateKind::Chinese)
            .unwrap();
        engine.commit(&candidate);
    };
    // kaif 下选 开发 两次,它升到第一
    pick(&mut engine, "kaif", "开发");
    pick(&mut engine, "kaif", "开发");
    engine.set_input("kaif");
    assert_eq!(texts_of(&engine)[0], "开发");
    assert_eq!(engine.learner().choice_balance("kaif", "开发"), 2);
    // 第三次选 开发 后删掉换 开放:开发 吃一笔负反馈(正分也退掉一份)
    pick(&mut engine, "kaif", "开发");
    engine.note_backspace();
    engine.note_backspace();
    pick(&mut engine, "kaif", "开放");
    // 净数 = +2 正(两次保留) − 1 负 = 1;关键是有负记录
    assert_eq!(engine.learner().choice_balance("kaif", "开发"), 1);
    // 下次查询:开放(有正分无负分)排到 开发 前面
    engine.set_input("kaif");
    assert_eq!(texts_of(&engine)[0], "开放");
}
