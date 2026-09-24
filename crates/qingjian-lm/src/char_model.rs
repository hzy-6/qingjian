//! 可映射的字符 n-gram 模型：给拼音约束的整句字束搜索打分。

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use fst::Map;
use memmap2::Mmap;
use qingjian_core::sentence::CharacterProposer;

use crate::LmError;

/// 字符 1–5 元计数，FST 常驻 mmap；概率公式与 `tools/eval/char_beam_probe.py` 对齐。
pub struct CharNgramModel {
    counts: Map<Mmap>,
    total: f64,
    vocabulary: f64,
}

impl CharacterProposer for CharNgramModel {
    fn propose(
        &self,
        choices: &[Vec<char>],
        beam_width: usize,
        limit: usize,
    ) -> Vec<(String, f64)> {
        Self::propose(self, choices, beam_width, limit)
    }

    fn score_text(&self, text: &str) -> f64 {
        Self::score_text(self, text)
    }
}

impl CharNgramModel {
    pub fn from_path(path: &Path) -> Result<Self, LmError> {
        let file = File::open(path)?;
        // SAFETY：生成的数据文件只整体替换，打开后不就地写入。
        let map = unsafe { Mmap::map(&file)? };
        let counts = Map::new(map)?;
        let total = counts
            .get("#TOTAL")
            .filter(|&value| value > 0)
            .ok_or(LmError::Corrupt("character model missing total count"))?
            as f64;
        let vocabulary = counts
            .get("#VOCAB")
            .filter(|&value| value > 0)
            .ok_or(LmError::Corrupt("character model missing vocabulary size"))?
            as f64;
        Ok(Self {
            counts,
            total,
            vocabulary,
        })
    }

    pub fn count(&self, gram: &str) -> u64 {
        self.counts.get(gram).unwrap_or(0)
    }

    fn count_cached(&self, gram: &str, cache: &mut HashMap<String, u64>) -> u64 {
        if let Some(&count) = cache.get(gram) {
            return count;
        }
        let count = self.count(gram);
        cache.insert(gram.to_owned(), count);
        count
    }

    /// 与离线探针相同的插值 log P(字 | 最多四个前字)。
    fn char_score(
        &self,
        prefix: &[char],
        character: char,
        cache: &mut HashMap<String, u64>,
    ) -> f64 {
        let weights = [0.03, 0.07, 0.15, 0.25, 0.5];
        let mut probabilities = Vec::with_capacity(5);
        let gram = character.to_string();
        let unigram =
            (self.count_cached(&gram, cache) as f64 + 0.1) / (self.total + 0.1 * self.vocabulary);
        probabilities.push(unigram);
        for order in 2..=5 {
            if prefix.len() < order - 1 {
                break;
            }
            let previous: String = prefix[prefix.len() - (order - 1)..].iter().collect();
            let mut gram = previous.clone();
            gram.push(character);
            probabilities.push(
                (self.count_cached(&gram, cache) as f64 + 0.1)
                    / (self.count_cached(&previous, cache) as f64 + 0.1 * self.vocabulary),
            );
        }
        let active = &weights[..probabilities.len()];
        let probability: f64 = active
            .iter()
            .zip(probabilities)
            .map(|(weight, probability)| weight * probability)
            .sum::<f64>()
            / active.iter().sum::<f64>();
        probability.ln()
    }

    pub fn score_text(&self, text: &str) -> f64 {
        let mut prefix = Vec::new();
        let mut score = 0.0;
        let mut cache = HashMap::new();
        for character in text.chars() {
            score += self.char_score(&prefix, character, &mut cache);
            prefix.push(character);
        }
        score
    }

    /// 每个音节仅用给定字表，按字 LM 取完整句子的前 `limit` 条。
    pub fn propose(
        &self,
        choices: &[Vec<char>],
        beam_width: usize,
        limit: usize,
    ) -> Vec<(String, f64)> {
        let mut beam: Vec<(f64, Vec<char>)> = vec![(0.0, Vec::new())];
        let mut cache = HashMap::new();
        for syllable_choices in choices {
            let mut expanded = Vec::with_capacity(beam.len() * syllable_choices.len());
            for (score, prefix) in &beam {
                for &character in syllable_choices {
                    let next_score = score + self.char_score(prefix, character, &mut cache);
                    let mut next = prefix.clone();
                    next.push(character);
                    expanded.push((next_score, next));
                }
            }
            if expanded.is_empty() {
                return Vec::new();
            }
            let compare = |a: &(f64, Vec<char>), b: &(f64, Vec<char>)| {
                b.0.total_cmp(&a.0).then_with(|| b.1.cmp(&a.1))
            };
            if expanded.len() > beam_width {
                expanded.select_nth_unstable_by(beam_width, compare);
            }
            let keep = expanded.len().min(beam_width);
            expanded[..keep].sort_unstable_by(compare);
            expanded.truncate(beam_width);
            beam = expanded;
        }
        beam.into_iter()
            .take(limit)
            .map(|(score, chars)| (chars.into_iter().collect(), score))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fst::MapBuilder;

    #[test]
    fn packed_counts_rank_plausible_pinyin_sentences() {
        let path = std::env::temp_dir().join(format!(
            "qingjian-char-model-{}-{}.fst",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut counts = [
            ("#TOTAL", 100),
            ("#VOCAB", 3),
            ("你", 50),
            ("你好", 20),
            ("你号", 1),
            ("号", 5),
            ("好", 30),
        ];
        counts.sort_by_key(|(gram, _)| *gram);
        let mut builder = MapBuilder::new(File::create(&path).unwrap()).unwrap();
        for (gram, count) in counts {
            builder.insert(gram, count).unwrap();
        }
        builder.finish().unwrap();

        let model = CharNgramModel::from_path(&path).unwrap();
        assert!(model.score_text("你好") > model.score_text("你号"));
        let choices = vec![vec!['你'], vec!['号', '好']];
        let proposals = model.propose(&choices, 2, 2);
        assert_eq!(proposals[0].0, "你好");
        assert_eq!(proposals[1].0, "你号");
        std::fs::remove_file(path).unwrap();
    }
}
