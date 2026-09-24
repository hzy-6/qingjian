//! 离线研究用：对 JSONL 中给定的候选文本直接取神经分，不走词图或引擎。

use std::path::Path;
#[cfg(feature = "qwen")]
use std::time::Instant;

#[cfg(feature = "qwen")]
use std::fs::File;
#[cfg(feature = "qwen")]
use std::io::{BufRead, BufReader};

#[cfg(feature = "qwen")]
use qingjian_core::sentence::SentenceScorer;
#[cfg(feature = "qwen")]
use serde::{Deserialize, Serialize};

use crate::{Args, CliError};

#[cfg(feature = "qwen")]
#[derive(Deserialize, Serialize)]
struct Row {
    context: String,
    texts: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scores: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    score_ms: Option<f64>,
}

pub(super) fn run(args: &Args, path: &Path) -> Result<(), CliError> {
    #[cfg(not(feature = "qwen"))]
    {
        let _ = (args, path);
        Err(CliError::Qwen(
            "--score-jsonl 需要 --features qwen".to_owned(),
        ))
    }
    #[cfg(feature = "qwen")]
    {
        let model = args.qwen.as_ref().expect("clap requires qwen");
        let scorer = qingjian_qwen::QwenScorer::load(model)
            .map_err(|error| CliError::Qwen(error.to_string()))?;
        for line in BufReader::new(File::open(path)?).lines() {
            let line = line?;
            let mut row: Row =
                serde_json::from_str(&line).map_err(|error| CliError::Qwen(error.to_string()))?;
            let texts: Vec<&str> = row.texts.iter().map(String::as_str).collect();
            let started = Instant::now();
            row.scores = Some(scorer.score(&row.context, &texts));
            row.score_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
            println!("{}", serde_json::to_string(&row).expect("row serializes"));
        }
        Ok(())
    }
}
