//! 拼写纠错 / 敲错边 / 模糊音。

use crate::correction::Edit;

use super::*;

/// 每个音节都合法、整句却不通的输入（`meiganxi`）：词图里的敲错边把 没关系 读出来，作为词候选插到最前；
/// 上屏按敲的字母消耗、记个人敲错表，退格重选时退回；原样说得通（`meiganxie` 没感谢）时不改。
#[test]
fn typo_edges_in_the_lattice_correct_legal_but_unlikely_pinyin() {
    let dictionary = Dictionary::parse(
        "没关系\tmei guan xi\t800000\n关系\tguan xi\t500000\n没\tmei\t900000\n干\tgan\t200000\n\
             洗\txi\t100000\n美感\tmei gan\t300000\n感谢\tgan xie\t300000\n谢\txie\t50000\n",
    )
    .unwrap();
    let mut engine =
        Engine::new(dictionary).with_learner(Box::new(CountingLearner(HashMap::new())));
    engine.set_input("meiganxi");
    let query = engine.query().unwrap();
    assert!(query.correction.is_none());
    let first = query.candidates.items[0].clone();
    assert_eq!(first.text, "没关系");
    assert_eq!(first.kind, CandidateKind::Chinese);
    assert_eq!(first.syllables, ["mei", "guan", "xi"]);
    // 词级候选不带敲错变体：原样的 美感 还在后面
    assert!(query.candidates.items.iter().any(|c| c.text == "美感"));
    assert_eq!(engine.commit(&first), "没关系");
    assert!(engine.composition().is_empty());
    assert_eq!(engine.learner().typo_count("gan", "guan"), 1);
    assert_eq!(engine.learner().choice_weight("meiganxi", "没关系"), 1);
    // 整个删掉、同一段拼音改选 美感：敲错的记录退回
    for _ in 0..3 {
        engine.note_backspace();
    }
    engine.set_input("meiganxi");
    let meigan = engine
        .query()
        .unwrap()
        .candidates
        .items
        .into_iter()
        .find(|c| c.text == "美感")
        .unwrap();
    engine.commit(&meigan);
    assert_eq!(engine.learner().typo_count("gan", "guan"), 0);
    assert_eq!(engine.composition().text(), "xi");
    // 原样读得通：没 + 感谢 是正常整句，不动
    engine.set_input("meiganxie");
    let query = engine.query().unwrap();
    assert_eq!(query.candidates.items[0].text, "没感谢");
    assert_eq!(query.candidates.items[0].kind, CandidateKind::Sentence);
    // 两个音节也纠：ganxi → 关系
    engine.set_input("ganxi");
    assert_eq!(engine.query().unwrap().candidates.items[0].text, "关系");
    // 太短不纠（不到 4 个字母）
    engine.set_input("gan");
    assert!(
        engine
            .query()
            .unwrap()
            .candidates
            .items
            .iter()
            .all(|c| c.text != "关系")
    );
    // 敲的拼音本身正好是一个词（按另一种切分）：不许敲错边压过它。jineng 最优切分是 jin eng，词图里读出 近藤，
    // 但 技能 的音节正好拼成整段输入
    let dictionary = Dictionary::parse(
        "技能\tji neng\t300000\n近藤\tjin teng\t900000\n近\tjin\t500000\n\
             机\tji\t400000\n能\tneng\t600000\n",
    )
    .unwrap();
    let mut engine = Engine::new(dictionary);
    engine.set_input("jineng");
    let query = engine.query().unwrap();
    assert_eq!(query.segmentations[0].joined("'"), "jin'eng");
    assert_eq!(query.candidates.items[0].text, "技能");
    assert!(query.candidates.items.iter().all(|c| c.text != "近藤"));
    // 退回原样的路径时原样的整句照出：shude 词图里 是的（shu → shi 相邻键）赢，但 树德 正好拼成 shude，
    // 于是整句退回 属的
    let dictionary = Dictionary::parse(
            "树德\tshu de\t500\n是的\tshi de\t5000000\n属\tshu\t100000\n的\tde\t8000000\n是\tshi\t7000000\n",
        )
        .unwrap();
    let mut engine = Engine::new(dictionary);
    engine.set_input("shude");
    let query = engine.query().unwrap();
    assert_eq!(query.candidates.items[0].text, "属的");
    assert_eq!(query.candidates.items[0].kind, CandidateKind::Sentence);
    assert!(query.candidates.items.iter().all(|c| c.text != "是的"));
}

/// 模糊音命中的词按敲的字母消耗拼音（`zi` 对 `zhi`），不算敲错。
#[test]
fn fuzzy_hits_consume_the_typed_syllables() {
    let dictionary =
        Dictionary::parse("知识\tzhi shi\t500000\n只是\tzhi shi\t600000\n资\tzi\t1000\n").unwrap();
    let mut engine =
        Engine::new(dictionary).with_learner(Box::new(CountingLearner(HashMap::new())));
    engine.set_fuzzy(FuzzyRules {
        z_zh: true,
        ..FuzzyRules::default()
    });
    engine.set_input("zishi");
    let zhishi = engine
        .query()
        .unwrap()
        .candidates
        .items
        .into_iter()
        .find(|c| c.text == "知识")
        .unwrap();
    engine.commit(&zhishi);
    assert!(engine.composition().is_empty());
    assert_eq!(engine.learner().typo_count("zi", "zhi"), 0);
}

/// 接受整段一处编辑的纠正也记个人敲错表：敲的那段字母对纠正后的音节。
#[test]
fn accepted_whole_string_correction_feeds_the_typo_table() {
    let dictionary = Dictionary::parse(
            "你好吗\tni hao ma\t5000\n你好\tni hao\t9000\n你\tni\t90000\n好\thao\t80000\n吗\tma\t70000\n",
        )
        .unwrap();
    let mut engine =
        Engine::new(dictionary).with_learner(Box::new(CountingLearner(HashMap::new())));
    engine.set_input("nihooma");
    let query = engine.query().unwrap();
    assert_eq!(query.correction.as_ref().unwrap().corrected, "nihaoma");
    let first = query.candidates.items[0].clone();
    engine.commit(&first);
    assert_eq!(engine.learner().typo_count("hoo", "hao"), 1);
    // 只选到 你 就上屏：没吃到编辑处，不算接受
    engine.set_input("nihooma");
    let ni = engine
        .query()
        .unwrap()
        .candidates
        .items
        .into_iter()
        .find(|c| c.text == "你")
        .unwrap();
    engine.commit(&ni);
    assert_eq!(engine.learner().typo_count("hoo", "hao"), 1);
}

#[test]
fn spelling_correction_fixes_one_edit_and_learns_from_enter() {
    let dictionary = Dictionary::parse(
            "你好吗\tni hao ma\t5000\n你好\tni hao\t9000\n你\tni\t90000\n好\thao\t80000\n吗\tma\t70000\n\
             和\the\t50000\n何\the\t3000\n咯\tlo\t100\n你猴\tni hou\t10\n",
        )
        .unwrap();
    let mut engine = Engine::new(dictionary)
        .with_english(WordList::parse("hello\n").unwrap())
        .with_learner(Box::new(CountingLearner(HashMap::new())));
    // 换错一个字母：nihooma → nihaoma，候选来自纠正后的拼音，拼音行画出被改掉的 o
    engine.set_input("nihooma");
    let query = engine.query().unwrap();
    let correction = query.correction.clone().expect("corrected");
    assert_eq!(correction.corrected, "nihaoma");
    assert_eq!(query.candidates.items[0].text, "你好吗");
    let kinds: Vec<(String, MarkedKind)> = query
        .marked_segments()
        .iter()
        .map(|s| (s.text.clone(), s.kind))
        .collect();
    assert_eq!(
        kinds,
        [
            ("ni'h".to_owned(), MarkedKind::Typed),
            ("o".to_owned(), MarkedKind::Corrected),
            ("ao'ma".to_owned(), MarkedKind::Typed),
        ]
    );
    assert_eq!(query.marked_cursor(), 10);
    // 选纠正后的词吃掉整段原串，并按原输入串记选择
    let first = query.candidates.items[0].clone();
    assert_eq!(engine.commit(&first), "你好吗");
    assert!(engine.composition().is_empty());
    engine.set_input("nihooma");
    assert_eq!(engine.query().unwrap().candidates.items[0].text, "你好吗");
    // 相邻换位
    engine.set_input("nihoama");
    let query = engine.query().unwrap();
    assert_eq!(query.correction.as_ref().unwrap().corrected, "nihaoma");
    // 选前缀词 你好 只吃到 nihoa 之后：换位不改长度，剩 ma
    let nihao = query
        .candidates
        .items
        .iter()
        .find(|c| c.text == "你好")
        .cloned()
        .unwrap();
    engine.commit(&nihao);
    assert_eq!(engine.composition().text(), "ma");
    // 整段是英文词的不纠（hello 不会变成 和咯）
    engine.set_input("hello");
    assert!(engine.query().unwrap().correction.is_none());
    // 原样已经说得通的合法简拼不纠：nhao 是 你好 的简拼，纠成 nihao 得分一样、扣掉编辑代价就输了
    engine.set_input("nhao");
    assert!(engine.query().unwrap().correction.is_none());
    // 末尾单字母只试相邻换位：nihoa → nihao
    engine.set_input("nihoa");
    assert_eq!(
        engine.query().unwrap().correction.unwrap().corrected,
        "nihao"
    );
    // 短串不纠
    engine.set_input("nih");
    assert!(engine.query().unwrap().correction.is_none());
    // 回车原样上屏过的串以后不纠
    engine.set_input("nihooma");
    assert!(engine.query().unwrap().correction.is_some());
    assert_eq!(engine.take_raw(), "nihooma");
    engine.set_input("nihooma");
    assert!(engine.query().unwrap().correction.is_none());
}

/// 漏一个字母仍召回：`nhaoma`（你好吗 漏 i）按 n… hao ma 的简拼整句原样读得通，不标纠正；
/// `nihama`（漏 o）每个音节都合法，词图里 ha → hao 的 Missing 敲错边把 你好吗 读到首位。
#[test]
fn missing_letter_still_recalls_the_word() {
    let dictionary = Dictionary::parse(
        "你好吗\tni hao ma\t5000\n你好\tni hao\t9000\n你\tni\t90000\n好\thao\t80000\n吗\tma\t70000\n",
    )
    .unwrap();
    let mut engine = Engine::new(dictionary);
    engine.set_input("nhaoma");
    let query = engine.query().unwrap();
    assert_eq!(query.candidates.items[0].text, "你好吗");
    assert!(query.correction.is_none());
    engine.set_input("nihama");
    let query = engine.query().unwrap();
    assert_eq!(query.candidates.items[0].text, "你好吗");
    assert!(query.correction.is_none());
}

/// 漏字母漏到切不动：`youi`（友谊 you yi 漏了第二个 y）只能切出 you 加尾巴 i（i 起不了音节），
/// 整串一处编辑里补回 y（Insert）后首位出 友谊。
#[test]
fn missing_letter_breaking_segmentation_gets_the_insert_edit() {
    let dictionary =
        Dictionary::parse("友谊\tyou yi\t8000\n友\tyou\t50000\n谊\tyi\t30000\n").unwrap();
    let mut engine = Engine::new(dictionary);
    engine.set_input("youi");
    let query = engine.query().unwrap();
    let correction = query.correction.as_ref().expect("补回 y");
    assert_eq!(correction.corrected, "youyi");
    assert!(matches!(correction.edit, Edit::Insert { index: 3 }));
    assert_eq!(query.candidates.items[0].text, "友谊");
}

/// 相邻字母颠倒不只在句首：末音节里敲反的 `nihaoam`（ma → am）与 `jintain`（tian → tain）
/// 都走相邻换位的纠正，首位出完整词。
#[test]
fn transposition_deep_inside_the_input_corrects_too() {
    let dictionary = Dictionary::parse(
        "你好吗\tni hao ma\t5000\n你好\tni hao\t9000\n你\tni\t90000\n好\thao\t80000\n吗\tma\t70000\n\
         今天\tjin tian\t6000\n今\tjin\t50000\n天\ttian\t40000\n",
    )
    .unwrap();
    let mut engine = Engine::new(dictionary);
    engine.set_input("nihaoam");
    let query = engine.query().unwrap();
    assert_eq!(query.correction.as_ref().unwrap().corrected, "nihaoma");
    assert_eq!(query.candidates.items[0].text, "你好吗");
    engine.set_input("jintain");
    let query = engine.query().unwrap();
    assert_eq!(query.correction.as_ref().unwrap().corrected, "jintian");
    assert_eq!(query.candidates.items[0].text, "今天");
}

/// 合法声母简拼的召回：`nh` 出 你好、`nhm` 出 你好吗，都是原样简拼命中、不带纠正。
#[test]
fn initials_recall_abbreviated_words() {
    let dictionary = Dictionary::parse(
        "你好吗\tni hao ma\t5000\n你好\tni hao\t9000\n你\tni\t90000\n好\thao\t80000\n吗\tma\t70000\n",
    )
    .unwrap();
    let mut engine = Engine::new(dictionary);
    engine.set_input("nh");
    let query = engine.query().unwrap();
    assert_eq!(query.candidates.items[0].text, "你好");
    assert!(query.correction.is_none());
    engine.set_input("nhm");
    let query = engine.query().unwrap();
    assert_eq!(query.candidates.items[0].text, "你好吗");
    assert!(query.correction.is_none());
}

/// 末尾单字母的补全：`jintm`、`shenm`、`weishm` 的末尾 m 都是简拼位置，整句把整段读出来
/// （jintm 连 m 一起读成 今天+么），首位召回完整读法，且不标纠正——末尾没打完的部分留在词图里等下一键。
#[test]
fn trailing_single_letter_completes_words() {
    let dictionary = Dictionary::parse(
        "今天\tjin tian\t6000\n今\tjin\t50000\n天\ttian\t40000\n什么\tshen me\t7000\n什\tshen\t30000\n\
         么\tme\t20000\n为什么\twei shen me\t8000\n为\twei\t60000\n",
    )
    .unwrap();
    let mut engine = Engine::new(dictionary);
    for (input, word) in [("jintm", "今天么"), ("shenm", "什么"), ("weishm", "为什么")] {
        engine.set_input(input);
        let query = engine.query().unwrap();
        assert_eq!(query.candidates.items[0].text, word, "{input}");
        assert!(query.correction.is_none(), "{input}");
    }
}

/// 短输入不过度纠错：两三个字母不出纠正标记，候选就是原样拼音的召回，
/// 相邻键敲错的变体词（ni→mi、ma→na、ge→he）不混进来；切不动的尾巴只留着等下一键。
#[test]
fn short_inputs_recall_verbatim_without_edited_variants() {
    let dictionary = Dictionary::parse(
        "你\tni\t90000\n咪\tmi\t40000\n吗\tma\t80000\n拿\tna\t40000\n个\tge\t90000\n和\the\t40000\n",
    )
    .unwrap();
    let mut engine = Engine::new(dictionary);
    for (input, exact, adjacent) in [("ni", "你", "咪"), ("ma", "吗", "拿"), ("ge", "个", "和")]
    {
        engine.set_input(input);
        let query = engine.query().unwrap();
        assert_eq!(query.candidates.items[0].text, exact, "{input}");
        assert!(query.correction.is_none(), "{input}");
        assert!(
            !query.candidates.items.iter().any(|c| c.text == adjacent),
            "{input}"
        );
    }
    // 三个字母同样不纠：nii 只按 ni 出 你，尾巴 i 不拿编辑去凑
    engine.set_input("nii");
    let query = engine.query().unwrap();
    assert!(query.correction.is_none());
    assert_eq!(query.candidates.items[0].text, "你");
    assert_eq!(query.tail, "i");
}

#[test]
fn fuzzy_rules_add_homophones_behind_exact_hits() {
    // 词库里只有 kai fa 系列加一个 哈；敲 kaiha 没开 f/h 时只有前缀词 开（开哈 原样读得通，词图的敲错边翻不过它），
    // 开了就出 开发（模糊命中）且覆盖更多字母排第一
    let mut engine = Engine::new(Dictionary::parse(&format!("{SAMPLE}哈\tha\t50000\n")).unwrap());
    engine.set_input("kaiha");
    let before = texts_of(&engine);
    assert!(!before.contains(&"开发".to_owned()));
    assert!(before.contains(&"开".to_owned()));
    engine.set_fuzzy(FuzzyRules {
        f_h: true,
        ..FuzzyRules::default()
    });
    let after = texts_of(&engine);
    // ha 是前缀，换成 fa 前缀后 开放（词频更高）与 开发 都出，覆盖更多字母排在 开 前面
    assert_eq!(after[0], "开放");
    assert!(after.contains(&"开发".to_owned()));
    assert!(after.contains(&"开".to_owned()));
    // 整句转换走同一套写法：xiangkaiha → 想开发
    engine.set_input("xiangkaiha");
    let query = engine.query().unwrap();
    assert_eq!(query.candidates.items[0].text, "想开发");
    assert_eq!(query.candidates.items[0].kind, CandidateKind::Sentence);
    // 敲对的仍然优先：kaifa 第一位还是 开发，且不重复
    engine.set_input("kaifa");
    let all = texts_of(&engine);
    assert_eq!(all[0], "开发");
    assert_eq!(all.iter().filter(|t| *t == "开发").count(), 1);
}

/// 长句测试共用的词库：今天天气冷吗。冷 / 坑 同音节差好几个数量级的词频，
/// 好让双错候选扣完两次代价与额外惩罚还能赢过安全 margin。
fn sentence_dictionary() -> Dictionary {
    Dictionary::parse(
        "今天\tjin tian\t6000\n天气\ttian qi\t50000\n冷\tleng\t900000\n坑\tkeng\t10\n\
         吗\tma\t70000\n吧\tba\t50000\n你好吗\tni hao ma\t5000\n你好\tni hao\t9000\n\
         你\tni\t90000\n好\thao\t80000\n",
    )
    .unwrap()
}

/// 长句中间一处错误：`wi`（qi 敲成旁边的 w）把整段切弄坏了，错误不在句尾，
/// 整段一处编辑的纠错把整句读出来。
#[test]
fn single_error_mid_sentence_still_corrects() {
    let mut engine = Engine::new(sentence_dictionary());
    engine.set_input("jintiantianwilengma");
    let query = engine.query().unwrap();
    assert_eq!(
        query.correction.as_ref().expect("纠出 wi→qi").corrected,
        "jintiantianqilengma"
    );
    assert_eq!(query.candidates.items[0].text, "今天天气冷吗");
}

/// 原样能得到自然整句的长句不纠：没有不像话的切分就不进纠错，候选留原样的读法。
#[test]
fn well_formed_long_sentence_keeps_its_reading() {
    let mut engine = Engine::new(sentence_dictionary());
    engine.set_input("jintiantianqilengma");
    let query = engine.query().unwrap();
    assert!(query.correction.is_none());
    assert_eq!(query.candidates.items[0].text, "今天天气冷吗");
}

/// 长句中间两处错误（q→w、l→k，都在键盘邻位）联合纠正：单错只能修出 今天天气坑吗，
/// 双错候选扣两次纠错代价与第二处的额外惩罚后仍明显胜出，整句正确；
/// 上屏吃整段原串并给两处各记一对敲错，回车原样上屏过的串以后不纠。
#[test]
fn double_error_mid_sentence_corrects_both() {
    let mut engine =
        Engine::new(sentence_dictionary()).with_learner(Box::new(CountingLearner(HashMap::new())));
    engine.set_input("jintiantianwikengma");
    let query = engine.query().unwrap();
    let correction = query.correction.clone().expect("两处敲错一起修出来");
    assert_eq!(correction.corrected, "jintiantianqilengma");
    assert!(matches!(
        correction.edit,
        Edit::Substitute {
            index: 11,
            from: 'w'
        }
    ));
    assert_eq!(
        correction.second,
        Some(Edit::Substitute {
            index: 13,
            from: 'k'
        })
    );
    assert_eq!(query.candidates.items[0].text, "今天天气冷吗");
    // 上屏消耗整段原串，两处编辑各记一对 (敲的, 要的)
    let first = query.candidates.items[0].clone();
    assert_eq!(engine.commit(&first), "今天天气冷吗");
    assert!(engine.composition().is_empty());
    assert_eq!(engine.learner().typo_count("wi", "qi"), 1);
    assert_eq!(engine.learner().typo_count("keng", "leng"), 1);
    // 回车原样上屏过的串以后不纠
    engine.set_input("jintiantianwikengma");
    assert!(engine.query().unwrap().correction.is_some());
    assert_eq!(engine.take_raw(), "jintiantianwikengma");
    engine.set_input("jintiantianwikengma");
    assert!(engine.query().unwrap().correction.is_none());
}

/// 错误过多不乱猜：三处敲错（wi、leng→keng、ma→ba）两处编辑修不全，
/// 正确整句不该被任何候选冒充；修正最多叠两处编辑，宁可修出一个通顺但不同的句子。
#[test]
fn three_errors_do_not_produce_the_right_sentence() {
    let mut engine = Engine::new(sentence_dictionary());
    engine.set_input("jintiantianwikengba");
    let query = engine.query().unwrap();
    assert!(
        query
            .candidates
            .items
            .iter()
            .all(|c| c.text != "今天天气冷吗")
    );
    if let Some(correction) = &query.correction {
        assert_ne!(correction.corrected, "jintiantianqilengma");
        // 修正至多两处编辑
        assert!(correction.second.is_some());
    }
}

/// 长拼音（24 个字母，正抵整段纠错的上限）带两处错误：预算把工作量钉死，
/// 候选数量随长度线性而非指数涨，整句仍纠得出来。
#[test]
fn long_input_with_two_errors_stays_bounded() {
    let mut engine = Engine::new(sentence_dictionary());
    engine.set_input("nihaomajintiantianwikeng");
    let start = std::time::Instant::now();
    let query = engine.query().unwrap();
    let elapsed = start.elapsed();
    assert_eq!(
        query
            .correction
            .as_ref()
            .expect("长句双错也纠出来")
            .corrected,
        "nihaomajintiantianqileng"
    );
    assert_eq!(query.candidates.items[0].text, "你好吗今天天气冷");
    // 双错搜索的上限是 6144 次切分检查 + 192 次整句转换（每种子至多 12 次，占位音节的转换也计数），与字母数线性相关；
    // 真要是按变体做笛卡尔积，这个量级的输入早就到分钟级了。上限放宽到秒级只为防回归。
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "查询耗时 {elapsed:?}"
    );
}

/// 双拼开着时整段纠错整路关闭，两处编辑的联合纠错也一样不碰它。
#[test]
fn double_correction_never_touches_shuangpin() {
    let mut engine = xiaohe();
    engine.set_input("jintiantianwikengma");
    let query = engine.query().unwrap();
    assert!(query.correction.is_none());
}
