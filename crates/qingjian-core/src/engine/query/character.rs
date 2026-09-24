//! 词级路径与字符级整句提议的合池，供神经模型从约 16 条里预选回固定 8 条。
//!
//! 词级词图会漏掉「池内只差一个同音字」的候选（如 是/时、在/再、的/得）。这里对胜出切分
//! 的每个音节取词库里的同音字，用字符 n-gram 模型（[`sentence::CharacterProposer`]）按拼音约束
//! 解码整句字串，得到一批字级候选；它们与词级路径合池后由同步打分器（Qwen）的原始分预选回 `limit` 条。
//! 字级候选本身分数压在原池之下（`-8` nat 优先级），进池只是给重排器多几个可选项。

use qingjian_dictionary::{Dictionary, SyllablePattern};

use crate::engine::Engine;
use crate::sentence::{Conversion, SentenceWord};

/// 每个音节的同音字表最多取多少个（字级束搜索的候选来源）。
const CHARACTER_CHOICES: usize = 50;

/// 字级束搜索的束宽。
const CHARACTER_BEAM: usize = 128;

/// 字级束搜索一次最多产出多少条整句字串。
const CHARACTER_LIMIT: usize = 64;

/// 字级候选相对旧最优路径的分数折扣：让它排在原池之后，只当重排器的备选。
const CHARACTER_PRIORITY: f64 = 8.0;

impl Engine {
    /// 把赢家切分的词级路径与同音字级整句提议合池，再按神经原始分预选回 `limit` 条。
    /// 只在接了字符模型、池里有多条路径、且能对齐汉字数时才补字级候选；旧静态首选始终保留。
    pub(super) fn with_character_proposals(
        &self,
        mut paths: Vec<Conversion>,
        dictionaries: &[&Dictionary],
        limit: usize,
    ) -> Vec<Conversion> {
        let Some(proposer) = &self.character_proposer else {
            return paths;
        };
        if limit < 2 || paths.len() < 2 {
            return paths;
        }
        // 以汉字数等于音节数的词级路径为底：整句字串逐音节对应，字表从它的音节取。
        let Some(source) = paths
            .iter()
            .find(|path| path.text.chars().count() == path.syllables.len())
            .cloned()
        else {
            return paths;
        };
        let syllables = source.syllables.clone();
        if syllables.len() < 2 {
            return paths;
        }
        let mut choices = Vec::with_capacity(syllables.len());
        for syllable in &syllables {
            let pattern = [SyllablePattern {
                text: syllable,
                complete: true,
            }];
            let mut characters = Vec::<(char, u32)>::new();
            for dictionary in dictionaries {
                for hit in dictionary.lookup_exact(&pattern) {
                    let mut text = hit.text.chars();
                    let (Some(character), None) = (text.next(), text.next()) else {
                        continue;
                    };
                    if let Some((_, frequency)) = characters
                        .iter_mut()
                        .find(|(existing, _)| *existing == character)
                    {
                        *frequency = (*frequency).max(hit.frequency);
                    } else {
                        characters.push((character, hit.frequency));
                    }
                }
            }
            if characters.is_empty() {
                return paths;
            }
            characters.sort_by_key(|item| std::cmp::Reverse(item.1));
            choices.push(
                characters
                    .into_iter()
                    .take(CHARACTER_CHOICES)
                    .map(|(character, _)| character)
                    .collect::<Vec<_>>(),
            );
        }
        let suggested = proposer.propose(&choices, CHARACTER_BEAM, CHARACTER_LIMIT);
        let old_len = paths.len();
        let base = &paths[0];
        let score = base.score - CHARACTER_PRIORITY;
        let static_score = base.static_score - CHARACTER_PRIORITY;
        for (text, _) in suggested {
            if paths.len() >= old_len + limit {
                break;
            }
            if paths.iter().any(|path| path.text == text) {
                continue;
            }
            let words = text
                .chars()
                .zip(&syllables)
                .map(|(character, syllable)| SentenceWord {
                    text: character.to_string(),
                    syllables: vec![syllable.clone()],
                    placeholder: false,
                })
                .collect();
            paths.push(Conversion {
                text,
                syllables: syllables.clone(),
                words,
                score,
                static_score,
                penalty: 0.0,
            });
        }
        self.preselect_character_paths(paths, old_len, limit)
    }

    /// 从「词级 + 字符级」合池里预选回 `limit` 条：同步打分器当场拿原始分排序（并写进缓存，
    /// 免得 `rescore_paths` 再打一遍）；异步打分器分还没到时先整池保留（让 `rescore_paths` 去 want），
    /// 全部到齐后按缓存里的原始分预选。旧静态首选钉住。
    fn preselect_character_paths(
        &self,
        mut paths: Vec<Conversion>,
        old_len: usize,
        limit: usize,
    ) -> Vec<Conversion> {
        if paths.len() <= limit || old_len == 0 {
            return paths;
        }
        let (context, after) = self.rescoring_window();
        let texts: Vec<String> = paths.iter().map(|path| path.text.clone()).collect();
        // 同步打分器：当场打原始分，写进缓存，再由原始分预选。
        if let Some(scorer) = &self.sentence_scorer {
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let scores = scorer.score_with_after(&context, &after, &refs);
            if scores.len() != texts.len() {
                paths.truncate(limit);
                return paths;
            }
            let cache_context = format!("{context}\0{after}");
            let mut cache = self.neural_cache.borrow_mut();
            cache.ensure_context(&cache_context);
            for (text, score) in texts.iter().zip(&scores) {
                cache.insert(text, *score);
            }
            drop(cache);
            return select_by_scores(paths, &scores, limit);
        }
        // 异步打分器：分还没到就整池留着（外层 `rescore_paths` 会 want 它们送去后台），
        // 全部到齐后按缓存里的原始分预选，与同步路径一致。
        let cache_context = format!("{context}\0{after}");
        let cached: Option<Vec<f64>> = {
            let mut cache = self.neural_cache.borrow_mut();
            cache.ensure_context(&cache_context);
            texts.iter().map(|text| cache.get(text)).collect()
        };
        match cached {
            Some(scores) => select_by_scores(paths, &scores, limit),
            None => paths,
        }
    }
}

/// 从合池里按神经原始分挑 `limit` 条：旧最优（下标 0）钉在首位，其余按分降序补齐，
/// 结果按原下标升序交回，保持「词级在前、字级在后」的相对顺序。
fn select_by_scores(paths: Vec<Conversion>, scores: &[f64], limit: usize) -> Vec<Conversion> {
    let mut order: Vec<usize> = (1..paths.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    let mut selected = Vec::with_capacity(limit);
    selected.push(0usize);
    selected.extend(order.into_iter().take(limit.saturating_sub(1)));
    selected.sort_unstable();
    selected
        .into_iter()
        .map(|index| paths[index].clone())
        .collect()
}
