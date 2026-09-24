//! 联想：候选之外的异步补充。
//!
//! [`Predictor`] 由壳注入（网络实现在 `qingjian-predict`，Core 永远不联网），Engine 负责裁剪上下文、
//! 编号请求、校验云端词的拼音、丢弃过期结果。联想**不参与排序、不阻塞输入**；
//! 云端词到了只补进候选窗口第一页末尾几格（[`crate::CandidateLayout`]），前面的本地候选不挪。
//!
//! Engine 这一侧的实现：组句中发请求、收结果校验、Tab 接受整句；翻译选中文字也走这里。

mod cloud_word;
mod fuzzy;
mod kind;
mod policy;
mod predictor;
mod question;
mod request;
mod response;
mod script;
mod surrounding_text;

pub use cloud_word::CloudWord;
pub use fuzzy::{mismatch_count, tolerance};
pub use kind::PredictionKind;
pub use policy::PredictionPolicy;
pub use predictor::{NoPredictor, Predictor};
pub use question::restates_question;
pub use request::PredictionRequest;
pub use response::Prediction;
pub use script::translation_target;
pub use surrounding_text::SurroundingText;

use crate::sentence::SentenceWord;

/// 纠错变体预筛：语言模型不认识一步时的兜底分（与 viterbi 的 `UNKNOWN_LOG_PROB` 同值）。
const VARIANT_UNKNOWN_LOG_PROB: f64 = -30.0;

/// 纠错变体最多覆盖几个词的路径：再长组合爆炸、收益也低。
const MAX_VARIANT_WORDS: usize = 6;

/// 每个词位置最多试几个同音替代（静态分最高的几个）。
const VARIANTS_PER_WORD: usize = 4;

/// 纠错变体总数上限。
const VARIANT_LIMIT: usize = 8;

/// 纠错候选要比第一页参照高出多少 nat 才展示：词槽可以宽，纠错改的是「当前首选可能错」，留余量防噪声。
const CORRECTION_MARGIN: f64 = 2.0;

/// 本地整句联想的门槛：音节数不到这个不唤醒 2B（与短词门控同一条线）。
const LOCAL_MIN_SYLLABLES: usize = 3;

/// 本地联想的上下文窗口：前后各 100 字（壳按 [`Engine::prediction_context_window`] 读，这里再裁一次）。
const LOCAL_CONTEXT_CHARS: usize = 100;

/// 静态后继最多枚举多少条进合并池。
const PROPOSAL_STATIC_LIMIT: usize = 32;

/// 提议池最多多少条送 2B。整批（基线 + 接续 + 参照 + 深池 + 纠错）要压在几十条以内：
/// 一批几百条会让后台线程忙几秒，把重排挤到超时（实测 408 条 5.1 秒）。
const PROPOSAL_LIMIT: usize = 8;

/// 第一页参照最多取几条（进批次的都是要花前向的）。
const REF_LIMIT: usize = 5;

/// 深池从候选列表第几条开始：第一页（缺省 9 条）之后才算被静态排序埋没的。
const DEEP_POOL_START: usize = 9;

/// 深池最多取多少条送 2B。
const DEEP_POOL_LIMIT: usize = 8;

/// 个人 n-gram 的一次计数折算成多少次静态计数：用户自己的习惯优先。
const PERSONAL_PROPOSAL_BOOST: u32 = 8;

/// 接续词最多几个字：再长多半是短语，不像「下一个词」。
const MAX_PROPOSAL_CHARS: usize = 6;

/// 在飞的本地整句联想：提议已送后台打分，结果按序号配对。
pub(super) struct LocalSentenceRequest {
    pub(super) sequence: u64,
    pub(super) guess: String,
    pub(super) proposals: Vec<String>,

    /// 第一页可见候选文本，当参照：深池词要胜过它们才值得捞。
    pub(super) refs: Vec<String>,

    /// 深池候选（文本 + 音节），按池序；胜过参照的捞进词槽。
    pub(super) words: Vec<(String, Vec<String>)>,

    /// 同音纠错变体（文本 + 整段音节）；胜过参照加余量的捞进词槽。
    pub(super) corrections: Vec<(String, Vec<String>)>,
}

/// 深池：第一页之外、全是汉字、音节齐全的候选（文本 + 音节），按池序去重。
/// 深池：第一页之外、全是汉字、音节齐全的候选（文本 + 音节），按池序去重。
fn deep_pool(candidates: &[Candidate]) -> Vec<(String, Vec<String>)> {
    let mut seen = std::collections::HashSet::new();
    candidates
        .iter()
        .skip(DEEP_POOL_START)
        .filter(|candidate| {
            !candidate.syllables.is_empty()
                && candidate.text.chars().all(crate::sentence::is_han)
                && seen.insert(candidate.text.clone())
        })
        .take(DEEP_POOL_LIMIT)
        .map(|candidate| (candidate.text.clone(), candidate.syllables.clone()))
        .collect()
}

/// 把一批本地联想打分拆成结果。打分段落：`[0]` 基线（guess 本身）、之后依次是接续提议、
/// 第一页参照、深池词、纠错变体。接续要胜基线才出整句；深池词要胜过全部参照才捞进词槽
/// （按分排序，至多 `slots` 条——0 表示不要词槽）；纠错变体与 guess 等长，要高出基线
/// [`CORRECTION_MARGIN`] 才出。
pub(super) fn split_local_scores(
    scores: &[f64],
    proposals: &[String],
    refs: &[String],
    words: &[(String, Vec<String>)],
    corrections: &[(String, Vec<String>)],
    slots: usize,
    guess: &str,
) -> (Option<String>, Vec<CloudWord>) {
    let baseline = scores.first().copied().unwrap_or(f64::NEG_INFINITY);
    let mut index = 1usize;
    let proposals_end = (index + proposals.len()).min(scores.len());
    let sentence = scores[index..proposals_end]
        .iter()
        .zip(proposals)
        .filter(|(score, _)| **score > baseline)
        .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, word)| format!("{guess}{word}"));
    index = proposals_end;
    let refs_end = (index + refs.len()).min(scores.len());
    // 参照取「每字平均分」：候选长短不一，log 概率逐字累加，不归一的活长参照自然占便宜
    let best_ref = scores[index..refs_end]
        .iter()
        .zip(refs)
        .map(|(score, text)| score / per_char(text))
        .fold(f64::NEG_INFINITY, f64::max);
    index = refs_end;
    let words_end = (index + words.len()).min(scores.len());
    let mut deep: Vec<(f64, &(String, Vec<String>))> = scores[index..words_end]
        .iter()
        .zip(words)
        .filter(|(score, (text, _))| **score / per_char(text) > best_ref)
        .map(|(score, candidate)| (*score, candidate))
        .collect();
    index = words_end;
    // 纠错变体与 guess 等长，不用归一；要高出余量（改的是「当前首选可能错」，留余量防噪声）
    let mut best_correction = f64::NEG_INFINITY;
    for (score, candidate) in scores[index..].iter().zip(corrections) {
        best_correction = best_correction.max(*score);
        if *score > baseline + CORRECTION_MARGIN {
            deep.push((*score, candidate));
        }
    }
    tracing::debug!(
        baseline,
        best_ref,
        best_correction,
        deep = deep.len(),
        corrections = corrections.len(),
        "本地联想打分分段"
    );
    deep.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let words = deep
        .into_iter()
        .take(slots)
        .map(|(_, (text, syllables))| CloudWord::local(text.clone(), syllables.clone()))
        .collect();
    (sentence, words)
}

/// 每字平均分的分母：至少 1，免得空文本除零。
fn per_char(text: &str) -> f64 {
    (text.chars().count().max(1)) as f64
}

/// 接续提议要像样：全是汉字、不太长、不是把 guess 结尾重复一遍。
fn is_proposable(word: &str, guess: &str) -> bool {
    !word.is_empty()
        && word.chars().count() <= MAX_PROPOSAL_CHARS
        && word.chars().all(crate::sentence::is_han)
        && !guess.ends_with(word)
}

use super::*;

impl Engine {
    /// 是否接了会联想的 Predictor；壳据此决定要不要起轮询定时器、画云朵标识。
    pub fn prediction_enabled(&self) -> bool {
        self.predictor.is_enabled()
    }

    /// 云联想或本地联想任一开着：壳据此发请求、起轮询。云朵标识仍只看 [`Self::prediction_enabled`]。
    pub fn prediction_active(&self) -> bool {
        self.predictor.is_enabled() || self.local_prediction
    }

    /// 壳读应用前后文的窗口：云联想按它的策略，本地联想开着时取两者较大。
    /// 窗口上限还是在这里：壳给得再多，发往云端的仍按云端策略裁，本地按本地窗口裁。
    pub fn prediction_context_window(&self) -> (usize, usize) {
        let policy = self.predictor.policy();
        let local = if self.local_prediction {
            LOCAL_CONTEXT_CHARS
        } else {
            0
        };
        (policy.before.max(local), policy.after.max(local))
    }

    pub fn prediction_policy(&self) -> PredictionPolicy {
        self.predictor.policy()
    }

    /// 发一次联想请求，返回序号；没接 Predictor、私密输入中或拼音太短时不发，返回 `None`。
    ///
    /// 要的是「当前作用域拼音对应的词」和整句补全；`candidates` 是本地候选，只取前几个当提示。
    /// `surrounding` 是应用给的光标前后文本，在这里按观察窗口裁剪，壳给得再多也只发这么多；
    /// 应用给不出上下文就只靠拼音，本地输入历史不可靠、不用。上屏之后不联想：没有拼音约束的下文联想只是噪音。
    pub fn request_prediction(
        &mut self,
        surrounding: Option<SurroundingText>,
        candidates: &[Candidate],
    ) -> Option<u64> {
        if self.private || !self.prediction_active() {
            return None;
        }
        // 不发也要换序号：正在飞的旧结果对应的是上一个输入状态，回来了也不能显示
        self.prediction_sequence += 1;
        let policy = self.predictor.policy();
        let (raw_before, raw_after) = match surrounding {
            Some(text) => (text.before, text.after),
            None => (String::new(), String::new()),
        };
        let (before, after) = (
            take_last_chars(&raw_before, policy.before),
            take_first_chars(&raw_after, policy.after),
        );
        let scope = self.composition.scope();
        if self.english_mode
            || self.modes().is_expression(scope, self.zhuyin)
            || is_raw(scope, self.modes(), self.shuangpin, self.zhuyin)
        {
            return None;
        }
        // 问字模式：问题本身就是全部上下文，不带应用文本、不要整句、本地没有候选可提示
        let question = self.modes().is_question(scope, self.zhuyin);
        if question
            && shortcut::unicode_form(self.modes().question_body(scope, self.zhuyin)).is_some()
        {
            // 码点输入本地就能答，不问云端
            return None;
        }
        let (kind, pinyin_source, before, after) = if question {
            (
                PredictionKind::Question,
                self.modes().question_body(scope, self.zhuyin),
                String::new(),
                String::new(),
            )
        } else {
            (PredictionKind::Compose, scope, before, after)
        };
        // 双拼：问云端用的是解出来的全拼，不是敲的键
        let decoded = self.decode(pinyin_source);
        let pinyin_source: &str = decoded.as_ref().map_or(pinyin_source, |d| d.pinyin());
        let letters = pinyin_source.chars().filter(|c| *c != '\'').count();
        if letters < MIN_PREDICTION_LETTERS {
            return None;
        }
        let (pinyin, syllables, guess, conversion) = match segment_longest_prefix(pinyin_source) {
            Ok((segmentations, tail)) => {
                let conversion = self.local_conversion(&segmentations);
                let guess = conversion
                    .as_ref()
                    .map_or_else(String::new, |best| best.text.clone());
                (
                    query::join_marked(&segmentations, tail),
                    segmentations.first().map_or(0, |s| s.syllables.len()),
                    guess,
                    conversion,
                )
            }
            Err(_) => (pinyin_source.to_owned(), 0, String::new(), None),
        };
        if question {
            self.last_question_guess = guess.clone();
        }
        let request = PredictionRequest {
            sequence: self.prediction_sequence,
            kind,
            before,
            after,
            pinyin,
            letters: pinyin_source.replace('\'', ""),
            syllables,
            candidates: candidates
                .iter()
                .take(PREDICTION_CANDIDATE_HINTS)
                .map(|c| c.text.clone())
                .collect(),
            guess: guess.clone(),
            max_items: policy.max_items,
            want_sentence: policy.sentence && !question,
            text: String::new(),
            target_language: String::new(),
        };
        self.last_prediction_kind = kind;
        self.last_prediction_scope = scope.to_owned();
        // 本地整句联想：音节要过短词门控（≤2 音节不唤醒 2B，与重排同一条线），
        // 提议送后台打分，结果从 poll_prediction 出。云与本地互不妨碍。
        let mut local_active = false;
        if self.local_prediction
            && policy.sentence
            && !question
            && kind == PredictionKind::Compose
            && syllables >= LOCAL_MIN_SYLLABLES
        {
            let refs = candidates
                .iter()
                .take(REF_LIMIT)
                .map(|candidate| candidate.text.clone())
                .collect::<Vec<_>>();
            local_active = self.submit_local_sentence(
                &guess,
                &raw_before,
                &raw_after,
                &refs,
                deep_pool(candidates),
                conversion.as_ref(),
            );
        }
        if !self.predictor.is_enabled() && !local_active {
            // 云与本地都不接这一单：序号已换（作废在飞的旧结果），壳也不用等
            return None;
        }
        self.predictor.submit(request);
        Some(self.prediction_sequence)
    }

    /// 把本地整句联想的接续提议与深池候选送去后台打分。两者都为空、模型不在就返回 `false`。
    fn submit_local_sentence(
        &mut self,
        guess: &str,
        raw_before: &str,
        raw_after: &str,
        refs: &[String],
        deep: Vec<(String, Vec<String>)>,
        conversion: Option<&Conversion>,
    ) -> bool {
        let proposals = self.sentence_proposals(guess);
        let corrections = conversion
            .map(|best| self.correction_variants(best))
            .unwrap_or_default();
        if proposals.is_empty() && deep.is_empty() && corrections.is_empty() {
            return false;
        }
        let Some(worker) = &self.rescorer else {
            return false;
        };
        // 壳不一定给得出应用文本（Electron / 终端），退回本会话最近上屏的字，与重排一致
        let before: String = if raw_before.is_empty() {
            self.history.recent(LOCAL_CONTEXT_CHARS).to_owned()
        } else {
            take_last_chars(raw_before, LOCAL_CONTEXT_CHARS)
        };
        let mut texts =
            Vec::with_capacity(1 + proposals.len() + refs.len() + deep.len() + corrections.len());
        // 第 0 条是基线：不带续写的 guess。续写要胜过它才算数。
        texts.push(guess.to_owned());
        texts.extend(proposals.iter().map(|word| format!("{guess}{word}")));
        texts.extend(refs.iter().cloned());
        texts.extend(deep.iter().map(|(text, _)| text.clone()));
        texts.extend(corrections.iter().map(|(text, _)| text.clone()));
        worker.submit_predict(
            self.prediction_sequence,
            before,
            take_first_chars(raw_after, LOCAL_CONTEXT_CHARS),
            texts,
        );
        self.local_sentence = Some(LocalSentenceRequest {
            sequence: self.prediction_sequence,
            guess: guess.to_owned(),
            proposals,
            refs: refs.to_owned(),
            words: deep,
            corrections,
        });
        true
    }

    /// 同音纠错变体：把最优转换里的词逐个换成同音词，静态整句分预筛后取前若干条。
    /// 返回（变体文本, 整段音节）——音节与文本等长，进词槽时按云端词同款校验。
    fn correction_variants(&self, conversion: &Conversion) -> Vec<(String, Vec<String>)> {
        let words = &conversion.words;
        // 整段就是一个词时没有可换的位置（那种情况靠词槽的同音候选）；太长的路径组合爆炸，先不做。
        if words.len() < 2 || words.len() > MAX_VARIANT_WORDS {
            return Vec::new();
        }
        let mut ranked: Vec<(f64, String)> = Vec::new();
        for (index, word) in words.iter().enumerate() {
            if word.placeholder || word.syllables.is_empty() {
                continue;
            }
            let positions: Vec<Vec<qingjian_dictionary::SyllablePattern<'_>>> = word
                .syllables
                .iter()
                .map(|syllable| vec![qingjian_dictionary::SyllablePattern::complete(syllable)])
                .collect();
            for hit in self
                .lookup_exact_all(&positions)
                .iter()
                .take(VARIANTS_PER_WORD)
            {
                if hit.text == word.text {
                    continue;
                }
                let text: String = words
                    .iter()
                    .enumerate()
                    .map(|(position, item)| {
                        if position == index {
                            hit.text
                        } else {
                            item.text.as_str()
                        }
                    })
                    .collect();
                ranked.push((self.variant_static_score(words, index, hit.text), text));
            }
        }
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut seen = std::collections::HashSet::new();
        let mut variants = Vec::new();
        for (_, text) in ranked {
            if text == conversion.text || !seen.insert(text.clone()) {
                continue;
            }
            variants.push((text, conversion.syllables.clone()));
            if variants.len() >= VARIANT_LIMIT {
                break;
            }
        }
        tracing::debug!(
            words = words.len(),
            variants = variants.len(),
            "纠错变体已生成"
        );
        variants
    }

    /// 变体的静态整句分（只问静态 LM，不认识就兜底）：只用来在变体之间排序。
    fn variant_static_score(&self, words: &[SentenceWord], index: usize, replacement: &str) -> f64 {
        let mut score = 0.0;
        let mut previous: Option<&str> = None;
        for (position, word) in words.iter().enumerate() {
            let text = if position == index {
                replacement
            } else {
                word.text.as_str()
            };
            score += self
                .language_model
                .log_prob(previous, text)
                .unwrap_or(VARIANT_UNKNOWN_LOG_PROB);
            previous = Some(text);
        }
        score
    }

    /// 本地整句联想的接续提议：静态 LM 与个人 n-gram 的后继合并排序。
    fn sentence_proposals(&self, guess: &str) -> Vec<String> {
        if guess.is_empty() {
            return Vec::new();
        }
        let Some(clauses) = sentence::segment_text(guess, self.language_model.as_ref()) else {
            return Vec::new();
        };
        let Some(previous) = clauses.last().and_then(|words| words.last()) else {
            return Vec::new();
        };
        let mut merged = std::collections::HashMap::<String, u32>::new();
        for (word, count) in self
            .language_model
            .successors(previous, PROPOSAL_STATIC_LIMIT)
        {
            *merged.entry(word).or_default() += count;
        }
        if let Some(personal) = self.learner.user_ngram() {
            for (word, count) in personal.successors(previous) {
                *merged.entry(word).or_default() += count.saturating_mul(PERSONAL_PROPOSAL_BOOST);
            }
        }
        let mut ranked: Vec<(String, u32)> = merged
            .into_iter()
            .filter(|(word, _)| is_proposable(word, guess))
            .collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked
            .into_iter()
            .take(PROPOSAL_LIMIT)
            .map(|(word, _)| word)
            .collect()
    }

    /// 把应用里选中的一段文字交给云端翻译（壳里快捷键触发）：主要是汉字就译成学习语言，是外文（拉丁字母、假名）就译成中文
    /// （[`translation_target`]）。云联想关着、私密输入中、文字为空时不发，返回 `None`；
    /// 译文从 [`Self::poll_prediction`] 的 `sentence` 里出。不进学习、不动缓冲区。
    pub fn request_translation(&mut self, text: &str) -> Option<u64> {
        let text = text.trim();
        if !self.predictor.is_enabled() || self.private || text.is_empty() {
            return None;
        }
        self.prediction_sequence += 1;
        let request = PredictionRequest {
            sequence: self.prediction_sequence,
            kind: PredictionKind::Translate,
            before: String::new(),
            after: String::new(),
            pinyin: String::new(),
            letters: String::new(),
            syllables: 0,
            candidates: Vec::new(),
            guess: String::new(),
            max_items: 1,
            want_sentence: true,
            text: text.to_owned(),
            target_language: translation_target(text, self.translator.language())
                .code()
                .to_owned(),
        };
        self.last_prediction_kind = PredictionKind::Translate;
        self.predictor.submit(request);
        Some(self.prediction_sequence)
    }

    /// 作废正在飞的联想（用户清空了拼音、关掉了联想框）。
    pub fn cancel_prediction(&mut self) {
        self.prediction_sequence += 1;
        self.local_sentence = None;
    }

    /// 取回最近一次请求的结果；过期结果直接丢。没有就绪的结果返回 `None`，不阻塞。
    /// 结果可能是空的（模型没给出可用条目），壳据此停止等待。
    pub fn poll_prediction(&mut self) -> Option<Prediction> {
        let mut next = self.next_prediction();
        while let Some(mut prediction) = next.take() {
            if prediction.sequence == self.prediction_sequence {
                // 问字的答案、翻译的译文和敲的拼音本来就对不上，只有组句联想的云端词要校验
                if self.last_prediction_kind == PredictionKind::Question {
                    let guess = &self.last_question_guess;
                    prediction
                        .words
                        .retain(|word| !restates_question(&word.text, guess));
                } else if self.last_prediction_kind == PredictionKind::Compose
                    && !self.question_mode()
                {
                    if self.predictor.policy().slots == 0 {
                        // 用户不要云端词，只留整句补全
                        prediction.words.clear();
                    } else {
                        self.validate_cloud_words(&mut prediction.words);
                    }
                    if !prediction.is_empty() {
                        // 给过用户什么：紧接着的上屏说明接没接受（本地联想能不能替代云端的尺子）
                        self.logger.record(InputLogEntry::Prediction {
                            scope: self.last_prediction_scope.clone(),
                            words: prediction.words.iter().map(|w| w.text.clone()).collect(),
                            sentence: prediction.sentence.clone(),
                        });
                    }
                }
                if self.traditional
                    && self.last_prediction_kind != PredictionKind::Translate
                    && let Some(opencc) = &self.opencc
                {
                    for word in &mut prediction.words {
                        let traditional = opencc.convert(&word.text);
                        self.traditional_map
                            .borrow_mut()
                            .insert(traditional.clone(), word.text.clone());
                        word.text = traditional;
                    }
                    if let Some(sentence) = &mut prediction.sentence {
                        let traditional = opencc.convert(sentence);
                        self.traditional_map
                            .borrow_mut()
                            .insert(traditional.clone(), sentence.clone());
                        *sentence = traditional;
                    }
                }
                return Some(prediction);
            }
            tracing::debug!(
                sequence = prediction.sequence,
                current = self.prediction_sequence,
                "丢弃过期的联想结果"
            );
            next = self.next_prediction();
        }
        None
    }

    /// 下一条原始联想结果：本地整句补全优先（它先到），云端次之。
    fn next_prediction(&mut self) -> Option<Prediction> {
        if let Some(prediction) = self.take_local_sentence() {
            return Some(prediction);
        }
        self.predictor.poll()
    }

    /// 收本地整句联想的后台打分：基线是不带续写的 guess，最好的续写要胜过它才算数。
    fn take_local_sentence(&mut self) -> Option<Prediction> {
        let sequence = self.local_sentence.as_ref()?.sequence;
        let worker = self.rescorer.as_ref()?;
        while let Some(scored) = worker.poll_predict() {
            if scored.sequence != sequence {
                continue; // 被更新的请求顶掉的旧结果
            }
            let request = self.local_sentence.take()?;
            if scored.scores.len() != scored.texts.len() {
                return Some(Prediction {
                    sequence: scored.sequence,
                    words: Vec::new(),
                    sentence: None,
                });
            }
            let slots = self.predictor.policy().slots;
            let (sentence, words) = split_local_scores(
                &scored.scores,
                &request.proposals,
                &request.refs,
                &request.words,
                &request.corrections,
                slots,
                &request.guess,
            );
            tracing::debug!(
                guess = %request.guess,
                sentence = ?sentence,
                words = ?words.iter().map(|word| word.text.as_str()).collect::<Vec<_>>(),
                "本地联想结果"
            );
            return Some(Prediction {
                sequence: scored.sequence,
                words,
                sentence,
            });
        }
        None
    }

    /// 只留下拼音对得上的云端词：字数等于音节数、每个音节合法、全拼与用户敲的字母的编辑距离在容许范围内
    /// （简拼不算错，允许少量错字 / 漏字 / 多字，纠错就靠这个）。模型偶尔会给出根本不是这个拼音的词，这些不进候选。
    pub(super) fn validate_cloud_words(&self, words: &mut Vec<CloudWord>) {
        let decoded = self.decode(self.composition.scope());
        let typed = decoded
            .as_ref()
            .map_or(self.composition.scope(), |d| d.pinyin());
        let letters = typed.chars().filter(|c| *c != '\'').count();
        let allowed = tolerance(letters);
        words.retain(|word| {
            let fits = !word.syllables.is_empty()
                && word.text.chars().count() == word.syllables.len()
                && word.syllables.iter().all(|s| parser::is_syllable(s))
                && mismatch_count(typed, &word.syllables) <= allowed;
            if !fits {
                tracing::debug!(text = %word.text, syllables = ?word.syllables, "云端词与拼音不符，丢弃");
            }
            fits
        });
    }

    /// 用户接受一条整句补全：作用域内的拼音作废、句子上屏。句子没有拼音，记不了词频与用户词，
    /// 但按语言模型把它切成词（[`sentence::segment_text`]）逐条记进个人 n-gram，与选整句候选一样；
    /// 标点处断句，句尾是标点时之后的词按句首记。整句退格删光再重打时这些转移一并退回。
    pub fn accept_prediction(&mut self, text: &str) -> String {
        let traditional_text = text.to_owned();
        let original_text_owned;
        let text = if self.traditional {
            original_text_owned = self
                .traditional_map
                .borrow()
                .get(text)
                .cloned()
                .unwrap_or_else(|| text.to_owned());
            &original_text_owned
        } else {
            text
        };
        let (_, input) = self.whole_scope();
        self.apply_retraction(&input, text);
        self.recording.clear();
        let keys = self.composition.scope().to_owned();
        let log_id = self.log_commit(&keys, text, InputSource::CloudSentence);
        self.meter_commit(text, InputSource::CloudSentence, false);
        self.composition.drain_scope();
        self.punctuation.note_committed(text);
        self.history.record(text);
        match sentence::segment_text(text, &*self.language_model) {
            Some(clauses) => {
                let buffer_left = !self.composition.is_empty();
                let count = clauses.len();
                for (index, words) in clauses.iter().enumerate() {
                    // 第一句接着前面上屏的词；标点之后的各句从句首起
                    if index > 0 {
                        self.chain.reset();
                    }
                    for word in words {
                        self.record_word(word, &[], 1, false, buffer_left || index + 1 < count);
                    }
                }
                if text.chars().last().is_some_and(|c| !c.is_alphanumeric()) {
                    self.chain.reset();
                }
                tracing::debug!(clauses = count, "整句补全已记入个人 n-gram");
            }
            None => self.chain.reset(),
        }
        let commit = LastCommit {
            text: text.to_owned(),
            chars: traditional_text.chars().count(),
            input,
            chosen: None,
            canonical: String::new(),
            application: self.application.clone(),
            transitions: std::mem::take(&mut self.recording),
            typos: Vec::new(),
            erased: 0,
            log_id,
            phrase: None,
        };
        self.remember_commit(commit);
        traditional_text
    }

    /// 云端词学成用户词时用哪套音节。模型给的读音偶有错（我的 → wo di），错读音学进去以后只会按错读音出来，
    /// 还会把整句候选里正确的同文本词顶掉；用户敲的拼音能切成与字数相同的完整音节、每个又是对应字的读音时以敲的为准，
    /// 否则仍用模型的，但模型的读音里有词库不认「这个字读这个音」的就不学。
    pub(super) fn learned_syllables(&self, candidate: &Candidate) -> Option<Vec<String>> {
        let chars: Vec<String> = candidate.text.chars().map(String::from).collect();
        let decoded = self.decode(self.composition.scope());
        let typed = decoded
            .as_ref()
            .map_or(self.composition.scope(), |d| d.pinyin());
        if let Ok((segmentations, "")) = segment_longest_prefix(typed)
            && let Some(best) = segmentations.first()
            && best.syllables.len() == chars.len()
            && best
                .syllables
                .iter()
                .zip(&chars)
                .all(|(s, c)| s.complete && self.accepts_reading(c, &s.text))
        {
            return Some(best.syllables.iter().map(|s| s.text.clone()).collect());
        }
        let fits = candidate.syllables.len() == chars.len()
            && candidate
                .syllables
                .iter()
                .zip(&chars)
                .all(|(s, c)| self.accepts_reading(c, s));
        fits.then(|| candidate.syllables.clone())
    }

    /// 这个字按这个读音能不能接受：词库里这个字有这个读音；词库根本不认识这个字（生僻字）时没法核对，照单全收。
    fn accepts_reading(&self, ch: &str, syllable: &str) -> bool {
        self.char_reads(ch, syllable) || !parser::SYLLABLES.iter().any(|s| self.char_reads(ch, s))
    }

    /// 词库（含用户词）里这个字有没有这个读音。
    fn char_reads(&self, ch: &str, syllable: &str) -> bool {
        let pattern = [qingjian_dictionary::SyllablePattern {
            text: syllable,
            complete: true,
        }];
        self.all_dictionaries()
            .into_iter()
            .any(|d| d.lookup_exact(&pattern).iter().any(|hit| hit.text == ch))
    }
}
