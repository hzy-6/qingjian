//! 青简 CLI：Phase 1 的测试工具。
//!
//! 输入拼音，打印候选（词性 + 译文）和各阶段耗时；输入序号上屏并记入用户词频。
//! 不依赖任何平台 API，是 Core 的第一个「壳」。

mod args;
mod display;
mod error;
mod eval;
mod logging;
mod repl;
mod replay;
mod rescoring;
mod tuning;

use std::time::Instant;

use clap::Parser;
use qingjian_core::{EmojiTable, Engine, FuzzyRules, Language};
use qingjian_dictionary::{Dictionary, WordList};
use qingjian_learning::FrequencyLearner;
use qingjian_lm::{BigramModel, CharNgramModel};
use qingjian_platform::Config;
use qingjian_predict::CloudPredictor;
use qingjian_translate::Glossary;

use crate::args::Args;
use crate::error::CliError;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), CliError> {
    dotenvy::dotenv().ok();
    let args = Args::parse();
    let _log_guard = logging::init()?;

    let started = Instant::now();
    let mut engine = build_engine(&args)?;
    tracing::info!(total_ms = started.elapsed().as_millis(), "Engine 就绪");
    engine.set_english_mode(args.english_mode);
    engine.set_chinese_first(args.chinese_first);
    tuning::apply(&mut engine, &args.tune)?;
    if let Some(path) = &args.replay {
        let report = replay::run(&mut engine, path, args.misses)?;
        print!("{report}");
        return Ok(());
    }
    if !args.eval_text.is_empty() {
        let report = eval::run(
            &mut engine,
            &args.eval_text,
            args.eval_save.as_deref(),
            args.eval_jsonl.as_deref(),
            args.misses,
            args.local_prediction,
        )?;
        print!("{report}");
        return Ok(());
    }
    if args.inputs.is_empty() {
        repl::run(&mut engine, args.limit)?;
    } else {
        for input in &args.inputs {
            println!("> {input}");
            if args.typing {
                display::show_typing(&mut engine, input);
            } else {
                display::show(&mut engine, input, args.limit);
            }
        }
    }
    engine.learner_mut().flush();
    Ok(())
}

/// 按参数挑整句重打分的第二打分来源：`--qwen` 指定 Qwen GGUF，不给就没有。
fn neural_scorer(
    args: &Args,
) -> Result<Option<Box<dyn qingjian_core::sentence::SentenceScorer>>, CliError> {
    let Some(path) = &args.qwen else {
        return Ok(None);
    };
    #[cfg(feature = "qwen")]
    {
        let scorer =
            qingjian_qwen::QwenScorer::load(path).map_err(|e| CliError::Qwen(e.to_string()))?;
        Ok(Some(Box::new(scorer)))
    }
    #[cfg(not(feature = "qwen"))]
    {
        let _ = path;
        Err(CliError::Qwen(
            "--qwen 需要用 `--features qwen` 编译（cargo run -p qingjian-cli --features qwen）"
                .to_owned(),
        ))
    }
}

/// 组装 Engine：这是 Core 之外唯一知道具体 Translator / Learner 类型的地方。
fn build_engine(args: &Args) -> Result<Engine, CliError> {
    let language: Language = args
        .language
        .parse()
        .map_err(|_| CliError::Language(args.language.clone()))?;
    if language == Language::Chinese {
        return Err(CliError::Language(args.language.clone()));
    }
    let dict_path = args
        .dict
        .clone()
        .unwrap_or_else(|| args::default_data_file("dict.tsv"));
    let glossary_path = args
        .glossary
        .clone()
        .unwrap_or_else(|| args::default_data_file(&format!("glossary-{}.tsv", language.code())));

    let started = Instant::now();
    let dictionary = Dictionary::from_path(&dict_path)?;
    let dict_load = started.elapsed();
    let started = Instant::now();
    let glossary = Glossary::from_path(language, &glossary_path)?;
    let glossary_load = started.elapsed();
    let english_path = args.english.clone().or_else(|| {
        let path = args::default_data_file("english.tsv");
        path.is_file().then_some(path)
    });
    let started = Instant::now();
    let english = english_path.as_ref().map(WordList::from_path).transpose()?;
    let english_load = started.elapsed();
    let learner = match &args.user_dict {
        Some(path) => FrequencyLearner::from_path(path)?,
        None => FrequencyLearner::default(),
    };
    tracing::info!(
        dict = %dict_path.display(),
        entries = dictionary.len(),
        glossary = %glossary_path.display(),
        glosses = glossary.len(),
        english = english.as_ref().map_or(0, WordList::len),
        learned = learner.len(),
        dict_ms = dict_load.as_millis(),
        glossary_ms = glossary_load.as_millis(),
        english_ms = english_load.as_millis(),
        "加载完成"
    );
    let mut engine = Engine::new(dictionary)
        .with_translator(Box::new(glossary))
        .with_learner(Box::new(learner));
    if !args.extra_dict.is_empty() {
        let mut extras = Vec::new();
        for path in &args.extra_dict {
            let dictionary = Dictionary::from_path(path)?;
            tracing::info!(path = %path.display(), entries = dictionary.len(), "附加词库已加载");
            extras.push(dictionary);
        }
        engine.set_extra_dictionaries(extras);
    }
    // 英文候选的中文释义可选
    let zh_glossary = args::default_data_file("glossary-zh.tsv");
    if zh_glossary.is_file() {
        let glossary = Glossary::from_path(Language::Chinese, &zh_glossary)?;
        tracing::info!(glosses = glossary.len(), "英→中释义表已加载");
        engine = engine.with_english_translator(Box::new(glossary));
    }
    if let Some(words) = english {
        engine = engine.with_english(words);
    }
    // emoji 表随仓库提供（Unicode License）：中文表 + 英文表合成一张，一张都没有就不出 emoji 候选
    let mut emoji: Option<EmojiTable> = None;
    let started = Instant::now();
    for name in ["emoji-zh.tsv", "emoji-en.tsv"] {
        let path = std::path::PathBuf::from("assets/emoji").join(name);
        if !path.is_file() {
            continue;
        }
        let table = EmojiTable::from_path(&path)?;
        match &mut emoji {
            Some(all) => all.merge(table),
            None => emoji = Some(table),
        }
    }
    if let Some(table) = emoji {
        tracing::info!(
            words = table.len(),
            load_ms = started.elapsed().as_millis(),
            "emoji 表已加载"
        );
        engine = engine.with_emoji(table);
    }
    // 语言模型可选：没有就退化成一元词频整句；打包过的 lm.qj 优先
    let packed = args
        .lm
        .clone()
        .unwrap_or_else(|| std::path::PathBuf::from("data/generated/lm.qj"));
    let unigram = std::path::PathBuf::from("data/generated/lm-unigram.tsv");
    let bigram = std::path::PathBuf::from("data/generated/lm-bigram.tsv");
    if packed.is_file() || (unigram.is_file() && bigram.is_file()) {
        let started = Instant::now();
        let model = if packed.is_file() {
            BigramModel::from_path(&packed)?
        } else {
            BigramModel::from_paths(&unigram, &bigram)?
        };
        tracing::info!(
            words = model.word_count(),
            bigrams = model.bigram_count(),
            load_ms = started.elapsed().as_millis(),
            "语言模型已加载"
        );
        engine = engine.with_language_model(Box::new(model));
    }
    if let Some(path) = &args.char_model {
        let started = Instant::now();
        let model = CharNgramModel::from_path(path)?;
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            "字符级整句提议已启用"
        );
        engine.set_character_proposer(Some(Box::new(model)));
    }
    if let Some(scorer) = neural_scorer(args)? {
        let started = Instant::now();
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            weight = args.neural_weight.unwrap_or(qingjian_core::NEURAL_WEIGHT),
            "神经重打分已启用"
        );
        engine = if args.neural_async {
            engine.with_async_sentence_scorer(
                scorer,
                args.neural_weight,
                args.neural_margin,
                args.neural_context,
            )
        } else {
            engine.with_sentence_scorer(
                scorer,
                args.neural_weight,
                args.neural_margin,
                args.neural_context,
            )
        };
        engine.set_neural_max_adjustment(args.neural_max_adjustment);
        if let Some(paths) = args.neural_paths {
            engine.set_neural_paths(paths);
        }
    }
    if args.local_prediction {
        // 联想打分走后台工作线程（`submit_predict`），同步打分器没有工作线程，开了也不会发
        if !args.neural_async {
            tracing::warn!("--local-prediction 需要 --neural-async，否则本地联想不会发出");
        }
        engine.set_local_prediction(true);
    }
    let config_path = args
        .config
        .clone()
        .unwrap_or_else(args::default_config_file);
    let mut config = Config::load(&config_path)?;
    if args.predict {
        config.predict.enabled = true;
    }
    if !args.fuzzy.is_empty() {
        let mut rules = FuzzyRules::default();
        for name in &args.fuzzy {
            if name == "all" {
                rules = FuzzyRules::ALL;
            } else if !rules.enable(name) {
                tracing::warn!(name, "不认识的模糊音规则，忽略");
            }
        }
        config.fuzzy = rules;
    }
    if config.fuzzy.any() {
        tracing::info!(rules = ?config.fuzzy, "模糊音已启用");
    }
    engine.set_traditional_mode(config.general.traditional);
    engine.set_fuzzy(config.fuzzy);
    engine.set_mode_keys(config.shortcut.mode);
    if let Some(scheme) = &args.shuangpin {
        config.general.shuangpin = if scheme == "off" {
            String::new()
        } else {
            scheme.clone()
        };
    }
    if let Some(scheme) = config.general.shuangpin() {
        tracing::info!(%scheme, "双拼已启用");
    }
    if config.general.zhuyin {
        tracing::info!("大千注音已启用");
    }
    engine.set_shuangpin(config.general.shuangpin());
    engine.set_zhuyin_mode(config.general.zhuyin);
    if config.predict.enabled {
        let predictor = CloudPredictor::new(&config.predict)?;
        engine = engine.with_predictor(Box::new(predictor));
    }
    Ok(engine)
}
