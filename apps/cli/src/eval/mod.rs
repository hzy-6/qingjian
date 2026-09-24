//! 整句评测：拿用户自己写的中文文本，转成他会敲的全拼，冷启动喂给引擎，看整句转换能不能把原句还原出来。
//!
//! 与 `--replay` 不同，这把尺子不依赖输入日志里「当时选了什么」（那多半是当时的引擎自己的输出），
//! 只看原文；整句排序、语言模型的改动先在同一份句子集上比过再合。
//! 输入既可以是原始文本（一行一段，按标点切句、汉字转拼音），也可以是之前 `--eval-save` 冻结下来的
//! `句子\t拼音\t上文` 三列文件；后者保证不同时间、不同分支比的是同一份句子。
//! 每句独立：不上屏、不学习，只把这句在原文里的上文写进输入历史给整句转换用。

mod extract;
mod pair;
mod report;
mod transcribe;

use std::collections::HashSet;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use qingjian_core::Engine;

pub use report::Report;

use pair::Pair;
use transcribe::Transcriber;

/// 跑一遍评测集，返回报告；`save` 给了就把用到的句子集写成三列文件。
pub fn run(
    engine: &mut Engine,
    paths: &[PathBuf],
    save: Option<&Path>,
    jsonl: Option<&Path>,
    show_misses: usize,
    probe_correction: bool,
) -> Result<Report, EvalError> {
    let mut report = Report::default();
    let pairs = collect(engine, paths, &mut report)?;
    if let Some(path) = save {
        let mut text = String::new();
        for pair in &pairs {
            text.push_str(&pair.to_line());
            text.push('\n');
        }
        std::fs::write(path, text).map_err(|source| EvalError::Write {
            path: path.to_owned(),
            source,
        })?;
        tracing::info!(path = %path.display(), count = pairs.len(), "句子集已保存");
    }
    let jsonl_target = jsonl.map(Path::to_owned);
    let mut jsonl = jsonl
        .map(|path| {
            std::fs::File::create(path)
                .map(BufWriter::new)
                .map_err(|source| EvalError::Write {
                    path: path.to_owned(),
                    source,
                })
        })
        .transpose()?;
    for pair in &pairs {
        if let Some(record) = evaluate(engine, pair, &mut report, show_misses, probe_correction)
            && let Some(writer) = &mut jsonl
        {
            serde_json::to_writer(&mut *writer, &record).map_err(EvalError::Json)?;
            writer.write_all(b"\n").map_err(|source| EvalError::Write {
                path: jsonl_target.clone().unwrap_or_default(),
                source,
            })?;
        }
    }
    if let Some(writer) = &mut jsonl {
        writer.flush().map_err(|source| EvalError::Write {
            path: jsonl_target.unwrap_or_default(),
            source,
        })?;
    }
    Ok(report)
}

#[derive(Debug, serde::Serialize)]
struct DistillRecord {
    gold: String,
    pinyin: String,
    context: String,
    candidates: Vec<String>,
    gold_rank: Option<usize>,
    rerank_pool: Vec<String>,
}

/// 读全部文件，得到去重后的句子集：有制表符的文件按冻结格式读，其余当原始文本抽句、转拼音。
fn collect(
    engine: &Engine,
    paths: &[PathBuf],
    report: &mut Report,
) -> Result<Vec<Pair>, EvalError> {
    let mut transcriber: Option<Transcriber> = None;
    let mut seen: HashSet<String> = HashSet::new();
    let mut pairs = Vec::new();
    for path in paths {
        let text = std::fs::read_to_string(path).map_err(|source| EvalError::Read {
            path: path.clone(),
            source,
        })?;
        if text.contains('\t') {
            for line in text.lines() {
                if let Some(pair) = Pair::parse(line)
                    && seen.insert(pair.text.clone())
                {
                    pairs.push(pair);
                }
            }
            continue;
        }
        let transcriber = transcriber.get_or_insert_with(|| {
            let dictionaries =
                std::iter::once(engine.dictionary()).chain(engine.extra_dictionaries());
            let transcriber = Transcriber::new(dictionaries);
            tracing::info!(words = transcriber.len(), "读音反查表已建");
            transcriber
        });
        for extracted in extract::extract(&text) {
            if !seen.insert(extracted.text.clone()) {
                continue;
            }
            report.extracted += 1;
            match transcriber.transcribe(&extracted.text, engine.language_model()) {
                Some(pinyin) => pairs.push(Pair {
                    text: extracted.text,
                    pinyin,
                    context: extracted.context,
                }),
                None => report.untranscribable += 1,
            }
        }
    }
    Ok(pairs)
}

/// 评一句：清空引擎状态、写入上文、喂拼音、看候选。
/// 上文只喂 Qwen(经 history):静态 bigram 是按句切分统计的,跨句左词是分布外输入,
/// 实测喂给静态反而 −0.5 个点(895 句混域尺),`Context::START` 就是正确的边界先验。
fn evaluate(
    engine: &mut Engine,
    pair: &Pair,
    report: &mut Report,
    show_misses: usize,
    probe_correction: bool,
) -> Option<DistillRecord> {
    report.total += 1;
    engine.clear();
    engine.break_chain();
    engine.history_mut().clear();
    engine.history_mut().record(&pair.context);
    engine.set_input(&pair.pinyin);
    let started = Instant::now();
    let query = match engine.query() {
        Ok(query) => query,
        Err(_) => {
            report.unparsable += 1;
            engine.clear();
            return None;
        }
    };
    // 异步重打分：像壳一样停顿后请求、等结果、再查一次；等的时间也算进查询耗时
    let query = if crate::rescoring::settle(engine) {
        engine.query().unwrap_or(query)
    } else {
        query
    };
    let elapsed = started.elapsed();
    report.query_time += elapsed;
    report.slowest_query = report.slowest_query.max(elapsed);
    // 重排探针:正确句有没有送进模型、被往哪个方向翻
    // 必须先读：下面的本地联想探针会再跑一次整句转换（`local_conversion`），而转换内部会写重排探针
    let probe = engine.rerank_probe();
    if let Some(probe) = &probe {
        report.probed += 1;
        if let Some(rank) = probe.pool.iter().position(|text| text == &pair.text) {
            report.oracle_hit += 1;
            report.oracle_rank_sum += rank + 1;
        }
        let before_right = probe.top_before.as_deref() == Some(pair.text.as_str());
        let after_right = probe.top_after.as_deref() == Some(pair.text.as_str());
        if before_right {
            report.static_top1 += 1;
            if !after_right {
                report.harmful_flips += 1;
            }
        } else if after_right {
            report.converted += 1;
        }
    }
    // 本地联想 / 同音纠错：这一句跑一次联想，看词槽里有没有原句（不改变候选排序，单独一把尺）
    if probe_correction {
        measure_correction(engine, pair, report, &query.candidates.items);
    }
    let items = &query.candidates.items;
    let position = items.iter().position(|c| c.text == pair.text);
    if position == Some(0) {
        report.top1 += 1;
    }
    // 第一个盖住全部拼音的候选就是整句转换的答案（整句本身是个词时也可能是词库词）
    let length = pair.text.chars().count();
    let sentence = items.iter().find(|c| c.text.chars().count() == length);
    report.chars_total += length;
    if let Some(sentence) = sentence {
        if sentence.text == pair.text {
            report.sentence_hit += 1;
        }
        report.chars_correct += sentence
            .text
            .chars()
            .zip(pair.text.chars())
            .filter(|(a, b)| a == b)
            .count();
    }
    if position != Some(0) && report.misses.len() < show_misses {
        let top: Vec<&str> = items.iter().take(3).map(|c| c.text.as_str()).collect();
        report.misses.push(format!(
            "{:<20} {:<28} 现在前三 {}{}",
            pair.text,
            pair.pinyin,
            top.join(" / "),
            position.map_or(String::from("（不在候选里）"), |i| format!(
                "（第 {} 位）",
                i + 1
            )),
        ));
    }
    let mut candidates = probe
        .as_ref()
        .map(|probe| probe.ranked.clone())
        .unwrap_or_default();
    if candidates.is_empty() {
        candidates.extend(
            items
                .iter()
                .filter(|candidate| candidate.text.chars().count() == length)
                .map(|candidate| candidate.text.clone()),
        );
        candidates.dedup();
        candidates.truncate(16);
    }
    let gold_rank = candidates
        .iter()
        .position(|candidate| candidate == &pair.text)
        .map(|rank| rank + 1);
    let record = DistillRecord {
        gold: pair.text.clone(),
        pinyin: pair.pinyin.clone(),
        context: pair.context.clone(),
        candidates,
        gold_rank,
        rerank_pool: probe.map_or_else(Vec::new, |probe| probe.pool),
    };
    engine.clear();
    Some(record)
}

/// 本地联想 / 同音纠错探针：这一句发一次联想请求，看词槽（纠错候选 / 深池词）里有没有原句。
/// 与重排探针分开：联想不参与候选排序，这是一把独立的尺。
fn measure_correction(
    engine: &mut Engine,
    pair: &Pair,
    report: &mut Report,
    candidates: &[qingjian_core::Candidate],
) {
    if engine.request_prediction(None, candidates).is_none() {
        return; // 音节太短 / 没模型 / 没开本地联想：这句不进分母
    }
    let started = Instant::now();
    let prediction = loop {
        if let Some(prediction) = engine.poll_prediction() {
            break prediction;
        }
        if started.elapsed() > std::time::Duration::from_secs(5) {
            report.correction_timeout += 1;
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    };
    report.correction_probed += 1;
    tracing::debug!(
        expected = %pair.text,
        pinyin = %pair.pinyin,
        got = ?prediction.words.iter().map(|word| word.text.as_str()).collect::<Vec<_>>(),
        sentence = ?prediction.sentence,
        "本地联想结果"
    );
    let hit = prediction.words.iter().any(|word| word.text == pair.text);
    if hit {
        report.correction_hit += 1;
    }
    let top1_right = candidates.first().is_some_and(|c| c.text == pair.text);
    if hit && !top1_right {
        report.correction_rescued += 1;
    }
    if top1_right && !prediction.words.is_empty() {
        report.correction_noise += 1;
    }
}

/// 整句评测的错误。
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("cannot read evaluation text {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("cannot write sentence set {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("cannot encode evaluation JSONL: {0}")]
    Json(#[source] serde_json::Error),
}
