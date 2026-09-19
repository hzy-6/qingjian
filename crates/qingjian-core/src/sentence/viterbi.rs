//! 词图上的最优路径：bigram Viterbi + 束搜索。
//!
//! 状态只按前一个词分（束宽内），个人三元要的前二词取前驱节点的回指（它那条最优路径上的前一个词）：
//! 不扩状态，代价是三元上下文是近似的，个人数据量下够用。

use qingjian_dictionary::{Dictionary, Match, SyllablePattern};

use super::{
    ABBREVIATED_SPAN_CANDIDATES, BEAM_WIDTH, Context, Conversion, LanguageModel,
    MAX_WORD_SYLLABLES, MIN_PARTIAL_LETTERS, Personal, SPAN_CANDIDATES, SentenceWord, SpanCache,
    SpanWord, fallback_log_prob, transition_log_prob,
};
use crate::ranking::weight_bonus;

/// 词库里没有的孤立音节（罕见音节没有单字）按这个 log 概率兜底，让路径总能走通。
const UNKNOWN_LOG_PROB: f64 = -30.0;

/// 一条部分路径的末尾节点。
struct Node {
    /// 这个词从第几个音节开始。
    start: usize,

    /// 词。
    text: String,

    /// 词的音节。
    syllables: Vec<String>,

    /// 到此为止的累计得分。
    score: f64,

    /// 累计得分里静态模型的部分（见 `Conversion::static_score`）。
    static_score: f64,

    /// 前驱在 `nodes[start]` 里的下标；`start == 0` 时无意义。
    back: usize,

    /// 次优前驱的下标与它的累计分（多数位置与最优同一条，末位分歧路径用）：
    /// 只按结尾词取 k 条时，前几名常是同一主干只换最后一个字的近重复，
    /// 神经重排真正需要的「前缀不同」的那条（雨下得 vs 余下的）恰恰排不进来，这里补一条。
    back2: usize,
    score2: f64,
    static_score2: f64,
    penalty2: f64,

    /// 这个词自己的兜底 log 概率（词库词频算的），分歧路径重算后缀转移分时用。
    fallback: f64,

    /// 是占位音节。
    placeholder: bool,

    /// 到此为止路径上模糊音 / 敲错变体的代价之和（已从 `score` 里扣掉，另记一份给调用方判断路径是不是原样）。
    penalty: f64,
}

/// 把音节序列转成最可能的词序列。`positions` 每个位置是若干写法（第一种是敲的，其余是模糊音 / 敲错变体），
/// `cost(位置, 命中的音节)` 是那个位置命中这种写法要扣的分（敲的原样 0），
/// `weight` 是用户选择次数，`personal` 是个人 n-gram 与插值参数（没有个人数据就传 [`Personal::NONE`]），`cache` 是格子候选的缓存
/// （调用方保证它与词库、`weight`、`personal`、`cost` 一致，这些一变就清）。
///
/// 简拼位置（`w x q`）按前缀取词：每个格子的候选会多得多，由语言模型在路径上分辨。
/// 全拼句子只有一个完整音节时，末尾单字母先不参与整句；已有两个完整音节时让模型判断这个前缀。
pub fn convert(
    dictionaries: &[&Dictionary],
    positions: &[Vec<SyllablePattern<'_>>],
    model: &dyn LanguageModel,
    personal: Personal<'_>,
    weight: impl Fn(&str) -> u32,
    cost: impl Fn(usize, &str) -> f64,
    cache: &mut SpanCache,
) -> Option<Conversion> {
    convert_with(
        dictionaries,
        positions,
        false,
        model,
        personal,
        weight,
        cost,
        cache,
    )
}

/// 同 [`convert`]，但全拼句子末尾的单字母也当一个音节读（`huo z…` → 或者）：
/// 给「整段拼音读法」与别的读法比分用，比分要两边覆盖同样多的字母。
pub fn convert_whole(
    dictionaries: &[&Dictionary],
    positions: &[Vec<SyllablePattern<'_>>],
    model: &dyn LanguageModel,
    personal: Personal<'_>,
    weight: impl Fn(&str) -> u32,
    cost: impl Fn(usize, &str) -> f64,
    cache: &mut SpanCache,
) -> Option<Conversion> {
    convert_with(
        dictionaries,
        positions,
        true,
        model,
        personal,
        weight,
        cost,
        cache,
    )
}

/// [`convert`] 与 [`convert_whole`] 的共同实现，`keep_partial` 选哪种；只要最优的一条。
#[allow(clippy::too_many_arguments)]
pub fn convert_with(
    dictionaries: &[&Dictionary],
    positions: &[Vec<SyllablePattern<'_>>],
    keep_partial: bool,
    model: &dyn LanguageModel,
    personal: Personal<'_>,
    weight: impl Fn(&str) -> u32,
    cost: impl Fn(usize, &str) -> f64,
    cache: &mut SpanCache,
) -> Option<Conversion> {
    convert_paths(
        dictionaries,
        positions,
        keep_partial,
        1,
        SPAN_CANDIDATES,
        Context::START,
        model,
        personal,
        weight,
        cost,
        cache,
    )
    .into_iter()
    .next()
}

/// 得分最高的前 `k` 条路径（最多束宽条，按得分降序，文本相同的只留一条）：给重打分用。
/// `span_width` 是每个格子收多少候选词（宽格子让低频字词——「拂」在 fu 格第 7——也进词图，交给神经重排分辨；
/// 缓存键带宽度，两种宽度互不污染）。`start` 是句首的左侧上下文（真实上文的末词；评测与裸调用给
/// [`Context::START`]），第一个词的静态转移与个人 n-gram 按它算,不再默认句首。
#[allow(clippy::too_many_arguments)]
pub fn convert_paths(
    dictionaries: &[&Dictionary],
    positions: &[Vec<SyllablePattern<'_>>],
    keep_partial: bool,
    k: usize,
    span_width: usize,
    start_context: Context<'_>,
    model: &dyn LanguageModel,
    personal: Personal<'_>,
    weight: impl Fn(&str) -> u32,
    cost: impl Fn(usize, &str) -> f64,
    cache: &mut SpanCache,
) -> Vec<Conversion> {
    let Some((last, head)) = positions.split_last() else {
        return Vec::new();
    };
    let Some(&last) = last.first() else {
        return Vec::new();
    };
    let abbreviated_head = head.iter().any(|p| p.first().is_none_or(|t| !t.complete));
    let positions = if keep_partial
        || last.complete
        || abbreviated_head
        || head.len() >= 2
        || last.text.len() >= MIN_PARTIAL_LETTERS
    {
        positions
    } else {
        head
    };
    let n = positions.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    // 句首左侧上下文;循环里 start 是下标,先拷出来用
    let start_ctx = start_context;
    let total: f64 = dictionaries
        .iter()
        .map(|d| d.total_frequency() as f64)
        .sum::<f64>()
        .max(1.0);
    let log_total = total.ln();

    // nodes[i]：覆盖前 i 个音节、以某个词结尾的部分路径；nodes[0] 是虚拟起点
    let mut nodes: Vec<Vec<Node>> = (0..=n).map(|_| Vec::new()).collect();
    nodes[0].push(Node {
        start: 0,
        text: String::new(),
        syllables: Vec::new(),
        score: 0.0,
        static_score: 0.0,
        back: 0,
        back2: 0,
        score2: 0.0,
        static_score2: 0.0,
        penalty2: 0.0,
        fallback: 0.0,
        placeholder: false,
        penalty: 0.0,
    });
    for start in 0..n {
        prune(&mut nodes[start]);
        if nodes[start].is_empty() {
            continue;
        }
        let mut any = false;
        for end in start + 1..=n.min(start + MAX_WORD_SYLLABLES) {
            let span = &positions[start..end];
            let cache_key = format!("{}\u{1}{span_width}", SpanCache::key(span));
            let hits = cache.get_or_insert_with(cache_key, || {
                span_candidates(
                    dictionaries,
                    span,
                    start,
                    span_width,
                    personal,
                    &weight,
                    &cost,
                )
            });
            if hits.is_empty() {
                continue;
            }
            any = true;
            for hit in hits.iter() {
                let bonus = weight_bonus(weight(&hit.text));
                let fallback = fallback_log_prob(hit.frequency, log_total);
                let ((score, back), (score2, back2)) = best_predecessor(
                    &nodes, start, &hit.text, model, personal, fallback, start_ctx,
                );
                let previous = &nodes[start][back];
                let penalty = previous.penalty + hit.penalty;
                // 静态首词的转移也按真实左文(有则用之),与 score 的口径一致
                let static_step = model
                    .log_prob(
                        if start > 0 {
                            Some(previous.text.as_str())
                        } else {
                            start_ctx.previous
                        },
                        &hit.text,
                    )
                    .unwrap_or(fallback);
                let static_score = previous.static_score + static_step;
                let (previous2_penalty, previous2_static, static_step2) = {
                    let previous2 = &nodes[start][back2];
                    (
                        previous2.penalty,
                        previous2.static_score,
                        model
                            .log_prob(
                                if start > 0 {
                                    Some(previous2.text.as_str())
                                } else {
                                    start_ctx.previous
                                },
                                &hit.text,
                            )
                            .unwrap_or(fallback),
                    )
                };
                let static_score2 = previous2_static + static_step2;
                nodes[end].push(Node {
                    start,
                    text: hit.text.clone(),
                    syllables: hit.syllables.clone(),
                    score: score + bonus - hit.penalty,
                    static_score,
                    back,
                    back2,
                    score2: score2 + bonus - hit.penalty,
                    static_score2,
                    penalty2: previous2_penalty + hit.penalty,
                    fallback,
                    placeholder: false,
                    penalty,
                });
            }
        }
        // 这个音节连单字都查不到：用音节本身占位，别让整句断掉
        if !any {
            let text = positions[start][0].text;
            let ((score, back), _) = best_predecessor(
                &nodes,
                start,
                text,
                &NoModel,
                Personal::NONE,
                UNKNOWN_LOG_PROB,
                start_ctx,
            );
            let penalty = nodes[start][back].penalty;
            let static_score = nodes[start][back].static_score + UNKNOWN_LOG_PROB;
            nodes[start + 1].push(Node {
                start,
                text: text.to_owned(),
                syllables: vec![text.to_owned()],
                score,
                static_score,
                back,
                back2: back,
                score2: score,
                static_score2: static_score,
                penalty2: penalty,
                fallback: UNKNOWN_LOG_PROB,
                placeholder: true,
                penalty,
            });
        }
    }
    prune(&mut nodes[n]);
    // 候选 = 每个结尾词的最优链，再加链上任意节点换次优前驱的分歧链：
    // 只取最优链时前 k 条全是同一主干只换最后一个字的近重复，重排器无案可翻；
    // 分歧只在末位还够不着中段的单字之争（`需要[再|在]研究`、`应该[坐|做]几路`），所以链上每个节点都试。
    let mut endings: Vec<usize> = (0..nodes[n].len()).collect();
    endings.sort_by(|&a, &b| {
        nodes[n][b]
            .score
            .partial_cmp(&nodes[n][a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // 分歧候选只在 k > 1（接了重排器）时生成：它们把束宽剪掉的路径按全分复活，
    // 自己当首选不如束内最优稳（真实回放 -1.6 个点），交给重排器再判才有净收益。
    let mut candidates: Vec<Conversion> = Vec::new();
    for &index in &endings {
        candidates.push(backtrack(&nodes, n, index));
    }
    if k > 1 {
        for &index in &endings {
            let chain = best_chain_indices(&nodes, n, index);
            for depth in 1..chain.len() {
                if let Some(conversion) = diverged(&nodes, &chain, depth, model, personal, &weight)
                {
                    candidates.push(conversion);
                }
            }
        }
    }
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut paths: Vec<Conversion> = Vec::with_capacity(k.min(candidates.len()));
    for conversion in candidates {
        if paths.len() >= k {
            break;
        }
        if !paths.iter().any(|p| p.text == conversion.text) {
            paths.push(conversion);
        }
    }
    paths
}

/// `nodes[position][index]` 的最优链，从头到尾的 (位置, 节点下标)。
fn best_chain_indices(
    nodes: &[Vec<Node>],
    mut position: usize,
    mut index: usize,
) -> Vec<(usize, usize)> {
    let mut chain = Vec::new();
    while position > 0 {
        chain.push((position, index));
        let node = &nodes[position][index];
        position = node.start;
        index = node.back;
    }
    chain.push((position, index));
    chain.reverse();
    chain
}

/// 链上第 `depth` 个节点换成它的次优前驱后的整条路径：
/// 前缀取次优前驱自己的最优链，后缀沿用原链的词，但前词变了、转移分要逐词重算。
fn diverged(
    nodes: &[Vec<Node>],
    chain: &[(usize, usize)],
    depth: usize,
    model: &dyn LanguageModel,
    personal: Personal<'_>,
    weight: &impl Fn(&str) -> u32,
) -> Option<Conversion> {
    let (position, index) = chain.get(depth).copied()?;
    let node = &nodes[position].get(index)?;
    if node.back2 == node.back {
        return None;
    }
    let mut head = backtrack(nodes, node.start, node.back2);
    let mut words = std::mem::take(&mut head.words);
    let mut score = node.score2;
    let mut static_score = node.static_score2;
    let mut penalty = node.penalty2;
    let mut previous = words.last().map(|w| w.text.clone());
    words.push(SentenceWord {
        text: node.text.clone(),
        syllables: node.syllables.clone(),
        placeholder: node.placeholder,
    });
    for (j, &(dpos, didx)) in chain[depth + 1..].iter().enumerate() {
        // 占位音节在 DP 里走 NoModel + Personal::NONE，这里用真模型重算结果相同
        //（模型对拼音串返回 None → 同一兜底分，personal 对非汉字计 0）；个人表混入非汉字时两处要一起改
        let word = &nodes[dpos][didx];
        // 个人三元的 earlier 与 DP 同口径：取链上前驱自己的回指词（DP 的近似上下文）。
        // 用复活链的实际词会让复活路径拿到比 DP 更「顺」的个人上下文而系统性虚高（真实回放 -1.6 个点的根源）
        let (ppos, pidx) = chain[depth + j];
        let parent = &nodes[ppos][pidx];
        let dp_earlier = if ppos > 0 {
            Some(nodes[parent.start][parent.back].text.clone())
        } else {
            None
        };
        let context = Context {
            previous: previous.as_deref(),
            earlier: dp_earlier.as_deref(),
        };
        score += transition_log_prob(model, personal, context, &word.text, word.fallback)
            + weight_bonus(weight(&word.text));
        static_score += model
            .log_prob(previous.as_deref(), &word.text)
            .unwrap_or(word.fallback);
        // 词自己的代价 = 它的总罚 − 链上前驱的总罚（代价只来自词本身，与走哪条前缀无关）；
        // 分数与罚都要走这份增量：分数漏扣会让带敲错 / 模糊边的分歧候选虚高
        let (ppos, pidx) = chain[depth + j];
        let delta = word.penalty - nodes[ppos][pidx].penalty;
        penalty += delta;
        score -= delta;
        previous = Some(word.text.clone());
        words.push(SentenceWord {
            text: word.text.clone(),
            syllables: word.syllables.clone(),
            placeholder: word.placeholder,
        });
    }
    let mut text = String::new();
    let mut syllables = Vec::new();
    for word in &words {
        text.push_str(&word.text);
        syllables.extend(word.syllables.iter().cloned());
    }
    Some(Conversion {
        text,
        syllables,
        words,
        score,
        static_score,
        penalty,
    })
}

/// 从 `nodes[position][index]` 沿最优前驱回溯出整条路径。
fn backtrack(nodes: &[Vec<Node>], mut position: usize, mut index: usize) -> Conversion {
    let node = &nodes[position][index];
    let (score, static_score, penalty) = (node.score, node.static_score, node.penalty);
    let mut words: Vec<SentenceWord> = Vec::new();
    while position > 0 {
        let node = &nodes[position][index];
        words.push(SentenceWord {
            text: node.text.clone(),
            syllables: node.syllables.clone(),
            placeholder: node.placeholder,
        });
        position = node.start;
        index = node.back;
    }
    words.reverse();
    let mut text = String::new();
    let mut syllables = Vec::new();
    for word in &words {
        text.push_str(&word.text);
        syllables.extend(word.syllables.iter().cloned());
    }
    Conversion {
        text,
        syllables,
        words,
        score,
        static_score,
        penalty,
    }
}

/// 占位音节不问语言模型。
struct NoModel;

impl LanguageModel for NoModel {
    fn log_prob(&self, _previous: Option<&str>, _word: &str) -> Option<f64> {
        None
    }
}

/// 一个格子里的候选词：所有词库的精确命中，按词频（加用户选择次数与个人出现次数，替代写法命中的按代价打折）取前几个。
/// 个人次数只在这里保证用户常用的同音词进得了格子，不进路径打分（那是 n-gram 的事）；
/// 打折让敲错变体命中的词只在原样命中不够多时才进格子，而常用词（关系）即使打折也留得住。
/// 格子里有简拼位置时命中的是一大片不同读音的词，多留一些让语言模型去挑。
fn span_candidates(
    dictionaries: &[&Dictionary],
    span: &[Vec<SyllablePattern<'_>>],
    start: usize,
    limit: usize,
    personal: Personal<'_>,
    weight: &impl Fn(&str) -> u32,
    cost: &impl Fn(usize, &str) -> f64,
) -> Vec<SpanWord> {
    let alternatives = span.iter().any(|p| p.len() > 1);
    let penalty_of = |m: &Match<'_>| {
        if !alternatives {
            return 0.0;
        }
        m.syllables()
            .enumerate()
            .map(|(index, syllable)| cost(start + index, syllable))
            .sum::<f64>()
    };
    // 得分先算好再排：单字母简拼的格子能命中几千条，比较器里每次查两张表会让排序占掉十几毫秒
    let mut scored: Vec<(f64, f64, Match<'_>)> = dictionaries
        .iter()
        .flat_map(|d| d.lookup_exact_alt(span))
        .map(|m| {
            let seen = weight(m.text) + personal.count(m.text);
            let penalty = penalty_of(&m);
            let score = f64::from(m.frequency) * (1.0 + f64::from(seen)) * (-penalty).exp();
            (score, penalty, m)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.dedup_by(|a, b| a.2.text == b.2.text);
    let abbreviated = span.iter().any(|p| p.iter().any(|t| !t.complete));
    scored.truncate(if abbreviated {
        ABBREVIATED_SPAN_CANDIDATES
    } else {
        limit
    });
    scored
        .into_iter()
        .map(|(_, penalty, hit)| SpanWord {
            text: hit.text.to_owned(),
            syllables: hit.syllables().map(str::to_owned).collect(),
            frequency: hit.frequency,
            penalty,
        })
        .collect()
}

/// 在 `nodes[start]` 的前驱里挑让 `word` 得分最高的两条，返回 ((最优分, 下标), (次优分, 下标))。
/// 转移概率先问静态模型（不认识就用词库兜底值），再与个人 n-gram 插值；前二词是前驱自己的前驱（回指）。
fn best_predecessor(
    nodes: &[Vec<Node>],
    start: usize,
    word: &str,
    model: &dyn LanguageModel,
    personal: Personal<'_>,
    fallback: f64,
    start_ctx: Context<'_>,
) -> ((f64, usize), (f64, usize)) {
    let mut best = (f64::NEG_INFINITY, 0);
    let mut second = (f64::NEG_INFINITY, 0);
    for (index, previous) in nodes[start].iter().enumerate() {
        let context = if start == 0 {
            start_ctx
        } else {
            Context {
                previous: Some(previous.text.as_str()),
                earlier: (previous.start > 0)
                    .then(|| nodes[previous.start][previous.back].text.as_str()),
            }
        };
        let score = previous.score + transition_log_prob(model, personal, context, word, fallback);
        if score > best.0 {
            second = best;
            best = (score, index);
        } else if score > second.0 {
            second = (score, index);
        }
    }
    if second.0 == f64::NEG_INFINITY {
        second = best;
    }
    (best, second)
}

/// 按得分降序只留束宽条。
fn prune(nodes: &mut Vec<Node>) {
    nodes.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    nodes.truncate(BEAM_WIDTH);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sentence::{FALLBACK_PENALTY, NoLanguageModel, UserNgram};

    const SAMPLE: &str = "我\two\t900000\n想\txiang\t500000\n去\tqu\t400000\n吃\tchi\t300000\n饭\tfan\t200000\n\
        吃饭\tchi fan\t100000\n我想\two xiang\t600000\n翔\txiang\t3000\n区\tqu\t100000\n卧\two\t2000\n\
        开发\tkai fa\t9000\n开\tkai\t20000\n发\tfa\t30000\n开放\tkai fang\t20000\n";

    fn complete<'a>(syllables: &[&'a str]) -> Vec<Vec<SyllablePattern<'a>>> {
        syllables
            .iter()
            .map(|s| vec![SyllablePattern::complete(s)])
            .collect()
    }

    fn unigram(
        dictionary: &Dictionary,
        patterns: &[Vec<SyllablePattern<'_>>],
    ) -> Option<Conversion> {
        convert(
            &[dictionary],
            patterns,
            &NoLanguageModel,
            Personal::NONE,
            |_| 0,
            |_, _| 0.0,
            &mut SpanCache::default(),
        )
    }

    #[test]
    fn picks_common_words_over_characters() {
        let dictionary = Dictionary::parse(SAMPLE).unwrap();
        let patterns = complete(&["wo", "xiang", "qu", "chi", "fan"]);
        let conversion = unigram(&dictionary, &patterns).unwrap();
        assert_eq!(conversion.text, "我想去吃饭");
        assert_eq!(conversion.word_count(), 3); // 我想 / 去 / 吃饭
        assert_eq!(conversion.words[0].text, "我想");
        assert_eq!(conversion.words[0].syllables, ["wo", "xiang"]);
        assert_eq!(conversion.syllables.len(), 5);
    }

    #[test]
    fn user_weight_lifts_near_ties_but_is_capped() {
        let dictionary = Dictionary::parse(SAMPLE).unwrap();
        // wo qu 没有整词：卧 / 我 词频差 450 倍（log 差 6.1），加分封顶后选多少次都翻不过来
        let patterns = complete(&["wo", "qu"]);
        let lifted = |count: u32| {
            convert(
                &[&dictionary],
                &patterns,
                &NoLanguageModel,
                Personal::NONE,
                |t| if t == "卧" { count } else { 0 },
                |_, _| 0.0,
                &mut SpanCache::default(),
            )
            .unwrap()
            .text
        };
        assert_eq!(lifted(20), "我去");
        assert_eq!(lifted(500), "我去");
        // 去 / 区 差 4 倍（log 差 1.4）：选过十几次就翻过来
        let patterns = complete(&["wo", "qu"]);
        let lifted = |count: u32| {
            convert(
                &[&dictionary],
                &patterns,
                &NoLanguageModel,
                Personal::NONE,
                |t| if t == "区" { count } else { 0 },
                |_, _| 0.0,
                &mut SpanCache::default(),
            )
            .unwrap()
            .text
        };
        assert_eq!(lifted(2), "我去");
        assert_eq!(lifted(20), "我区");
    }

    #[test]
    fn partial_last_syllable_of_a_full_pinyin_sentence() {
        let dictionary = Dictionary::parse(SAMPLE).unwrap();
        let mut patterns = complete(&["wo", "xiang"]);
        patterns.push(vec![SyllablePattern::prefix("ka")]);
        assert_eq!(unigram(&dictionary, &patterns).unwrap().text, "我想开");
        // 两个完整音节后，末尾单字母也参与组句
        let mut patterns = complete(&["wo", "xiang"]);
        patterns.push(vec![SyllablePattern::prefix("k")]);
        assert_eq!(unigram(&dictionary, &patterns).unwrap().text, "我想开");
    }

    fn abbreviated<'a>(letters: &[&'a str]) -> Vec<Vec<SyllablePattern<'a>>> {
        letters
            .iter()
            .map(|s| vec![SyllablePattern::prefix(s)])
            .collect()
    }

    #[test]
    fn abbreviated_sentences_convert_by_prefix() {
        let dictionary = Dictionary::parse(SAMPLE).unwrap();
        // 全简拼：每个格子按前缀取词，末尾单字母也是一个音节
        let conversion = unigram(&dictionary, &abbreviated(&["w", "x", "q"])).unwrap();
        assert_eq!(conversion.text, "我想去");
        assert_eq!(conversion.syllables, ["wo", "xiang", "qu"]);
        assert_eq!(conversion.words[0].text, "我想");
        // 简拼与全拼混用
        let patterns = [
            vec![SyllablePattern::prefix("w")],
            vec![SyllablePattern::complete("xiang")],
            vec![SyllablePattern::prefix("q")],
            vec![SyllablePattern::prefix("c")],
            vec![SyllablePattern::prefix("f")],
        ];
        assert_eq!(unigram(&dictionary, &patterns).unwrap().text, "我想去吃饭");
        // 简拼位置按前缀取词：`f` 既是 fa 也是 fang，词频高的 开放 胜出
        let conversion = unigram(&dictionary, &abbreviated(&["k", "f"])).unwrap();
        assert_eq!(conversion.text, "开放");
        assert!(!conversion.has_placeholder());
    }

    /// 只认 `我 → 翔` 的假模型下，简拼 `w x` 也该听语言模型的。
    #[test]
    fn language_model_decides_abbreviated_homophones() {
        let dictionary = Dictionary::parse(SAMPLE).unwrap();
        let patterns = abbreviated(&["w", "x"]);
        let conversion = convert(
            &[&dictionary],
            &patterns,
            &XiangModel,
            Personal::NONE,
            |_| 0,
            |_, _| 0.0,
            &mut SpanCache::default(),
        )
        .unwrap();
        assert_eq!(conversion.text, "我翔");
    }

    #[test]
    fn unknown_syllables_are_kept_as_pinyin() {
        let dictionary = Dictionary::parse(SAMPLE).unwrap();
        let patterns = complete(&["wo", "zhuang", "qu"]);
        let conversion = unigram(&dictionary, &patterns).unwrap();
        assert_eq!(conversion.text, "我zhuang去");
        assert!(conversion.has_placeholder());
        assert!(conversion.words.iter().filter(|w| w.placeholder).count() == 1);
    }

    /// 只认 `我 → 翔` 的假模型：bigram 应该压过一元词频。
    struct XiangModel;

    impl LanguageModel for XiangModel {
        fn log_prob(&self, previous: Option<&str>, word: &str) -> Option<f64> {
            match (previous, word) {
                (Some("我"), "翔") => Some(-2.0),
                (Some("我"), "想") => Some(-8.0),
                (None, "我") => Some(-1.0),
                _ => None,
            }
        }
    }

    #[test]
    fn language_model_decides_between_homophones() {
        let dictionary = Dictionary::parse(SAMPLE).unwrap();
        let patterns = complete(&["wo", "xiang"]);
        let conversion = convert(
            &[&dictionary],
            &patterns,
            &XiangModel,
            Personal::NONE,
            |_| 0,
            |_, _| 0.0,
            &mut SpanCache::default(),
        )
        .unwrap();
        assert_eq!(conversion.text, "我翔");
        assert_eq!(conversion.word_count(), 2);
    }

    #[test]
    fn personal_ngram_overrides_static_model_after_two_selections() {
        let dictionary = Dictionary::parse(SAMPLE).unwrap();
        let patterns = complete(&["wo", "xiang"]);
        let mut personal = UserNgram::default();
        let text = |personal: &UserNgram| {
            convert(
                &[&dictionary],
                &patterns,
                &XiangModel,
                Personal::new(Some(personal)),
                |_| 0,
                |_, _| 0.0,
                &mut SpanCache::default(),
            )
            .unwrap()
            .text
        };
        assert_eq!(text(&personal), "我翔");
        // 静态模型给 翔 的是很强的 bigram（P ≈ 0.14）：用户选过一次 我 → 想 翻不过（防误选），两次就翻。
        // 静态证据越弱（P 越小），个人偏好翻过来得越早。
        personal.record(Context::START, "我");
        personal.record(Context::after("我"), "想");
        assert_eq!(text(&personal), "我翔");
        personal.record(Context::START, "我");
        personal.record(Context::after("我"), "想");
        assert_eq!(text(&personal), "我想");
    }

    /// 三元上下文来自前驱的回指：「我想」后面的 去 / 区 由用户在「我想」后选过什么决定，
    /// 而「卧想」后面（回指不同）拿不到这条三元。
    #[test]
    fn trigram_context_comes_from_the_predecessor_chain() {
        let dictionary = Dictionary::parse(SAMPLE).unwrap();
        let patterns = complete(&["wo", "xiang", "qu"]);
        let convert_with = |personal: &UserNgram| {
            convert(
                &[&dictionary],
                &patterns,
                &NoLanguageModel,
                Personal::new(Some(personal)),
                |_| 0,
                |_, _| 0.0,
                &mut SpanCache::default(),
            )
            .unwrap()
        };
        let mut personal = UserNgram::default();
        // 一元下 我想去（我想 是一个词）
        assert_eq!(convert_with(&personal).text, "我想去");
        // 用户在 我 → 想 之后选过 区：三元 (我, 想) → 区 把 区 抬过 去，路径改走 我 / 想 / 区
        for _ in 0..4 {
            personal.record(Context::START, "我");
            personal.record(Context::after("我"), "想");
            personal.record(Context::after_two("我", "想"), "区");
        }
        let conversion = convert_with(&personal);
        assert_eq!(conversion.text, "我想区");
        assert_eq!(conversion.word_count(), 3);
    }

    #[test]
    fn cached_spans_give_the_same_sentence_and_only_new_spans_are_computed() {
        let dictionary = Dictionary::parse(SAMPLE).unwrap();
        let mut cache = SpanCache::default();
        let run = |cache: &mut SpanCache, syllables: &[&str]| {
            let patterns = complete(syllables);
            convert(
                &[&dictionary],
                &patterns,
                &NoLanguageModel,
                Personal::NONE,
                |_| 0,
                |_, _| 0.0,
                cache,
            )
            .unwrap()
            .text
        };
        let fresh = run(&mut cache, &["wo", "xiang", "qu", "chi"]);
        let before = cache.len();
        // 多敲一个音节：只新增以它结尾的格子
        let extended = run(&mut cache, &["wo", "xiang", "qu", "chi", "fan"]);
        assert_eq!(extended, "我想去吃饭");
        assert!(cache.len() > before);
        assert!(cache.len() - before <= MAX_WORD_SYLLABLES);
        // 再算一遍全部命中缓存，结果一致
        let again = cache.len();
        assert_eq!(run(&mut cache, &["wo", "xiang", "qu", "chi"]), fresh);
        assert_eq!(cache.len(), again);
    }

    /// 敲错变体是带代价的边：`gan xi` 在 `gan` 位多一种写法 `guan`（代价 4.5），原样凑不出像样的句子时 关系 胜出，
    /// 原样本身说得通（`gan xie` 感谢）时代价让它输。
    #[test]
    fn typo_alternatives_are_penalized_edges() {
        let dictionary = Dictionary::parse(
            "关系\tguan xi\t500000\n干\tgan\t20000\n洗\txi\t10000\n感谢\tgan xie\t300000\n关\tguan\t30000\n谢\txie\t5000\n",
        )
        .unwrap();
        let cost = |index: usize, syllable: &str| {
            if index == 0 && syllable == "guan" {
                4.5
            } else {
                0.0
            }
        };
        let run = |positions: Vec<Vec<SyllablePattern<'_>>>| {
            convert(
                &[&dictionary],
                &positions,
                &NoLanguageModel,
                Personal::NONE,
                |_| 0,
                cost,
                &mut SpanCache::default(),
            )
            .unwrap()
        };
        let with_typo = |second: &'static str| {
            vec![
                vec![
                    SyllablePattern::complete("gan"),
                    SyllablePattern::complete("guan"),
                ],
                vec![SyllablePattern::complete(second)],
            ]
        };
        let conversion = run(with_typo("xi"));
        assert_eq!(conversion.text, "关系");
        assert_eq!(conversion.words[0].syllables, ["guan", "xi"]);
        // 原样 干洗 两个单字的得分远低于 关系 − 4.5：噪声信道选纠正，路径带着代价
        assert_eq!(conversion.penalty, 4.5);
        let conversion = run(with_typo("xie"));
        assert_eq!(conversion.text, "感谢");
        assert_eq!(conversion.penalty, 0.0);
    }

    /// 中段分歧路径：`xuyaozaiyanjiuyixia` 的最优链在研究前选了高频的 在，
    /// 次优的 再 是研究节点的次优前驱——分歧不在末位也要进得了候选，重排器才有案可翻。
    #[test]
    fn paths_include_mid_chain_divergence() {
        let dictionary = Dictionary::parse(
        "需要\txu yao\t90000\n在\tzai\t90000\n再\tzai\t30000\n研究\tyan jiu\t80000\n一下\tyi xia\t70000\n",
    )
    .unwrap();
        let patterns = complete(&["xu", "yao", "zai", "yan", "jiu", "yi", "xia"]);
        let paths = convert_paths(
            &[&dictionary],
            &patterns,
            false,
            8,
            SPAN_CANDIDATES,
            Context::START,
            &NoLanguageModel,
            Personal::NONE,
            |_| 0,
            |_, _| 0.0,
            &mut SpanCache::default(),
        );
        assert_eq!(paths[0].text, "需要在研究一下");
        assert!(
            paths.iter().any(|p| p.text == "需要再研究一下"),
            "paths: {:?}",
            paths.iter().map(|p| p.text.clone()).collect::<Vec<_>>()
        );
    }

    /// 带代价边的对拍：分歧候选的后缀含敲错 / 模糊边命中时，重算的分数必须把这些代价都扣掉。
    /// 不变量（NoModel、无个人、无用户加分）：每条候选 score + penalty == 各词兜底分之和。
    /// 这里曾漏扣后缀词代价（`大打关系` 的分歧在 打 处、后缀 关系 带 0.7 罚，分数虚高恰好 0.7），用这条钉住。
    #[test]
    fn diverged_candidates_carry_their_suffix_penalties() {
        let dictionary = Dictionary::parse(
        "想\txiang\t50000\n向\txiang\t40000\n大\tda\t30000\n打\tda\t40000\n关系\tguan xi\t300000\n干\tgan\t20000\n洗\txi\t10000\n吧\tba\t80000\n",
    )
    .unwrap();
        let log_total = (dictionary.total_frequency() as f64).max(1.0).ln();
        let patterns = vec![
            vec![SyllablePattern::complete("xiang")],
            vec![SyllablePattern::complete("da")],
            vec![
                SyllablePattern::complete("gan"),
                SyllablePattern::complete("guan"),
            ],
            vec![SyllablePattern::complete("xi")],
            vec![SyllablePattern::complete("ba")],
        ];
        let paths = convert_paths(
            &[&dictionary],
            &patterns,
            false,
            16,
            SPAN_CANDIDATES,
            Context::START,
            &NoLanguageModel,
            Personal::NONE,
            |_| 0,
            |index, syllable| match (index, syllable) {
                (2, "guan") => 0.7,
                _ => 0.0,
            },
            &mut SpanCache::default(),
        );
        // 先确认用例真的产出了后缀带代价的分歧候选（打 处分歧 → 头换 大，后缀 关系 带 0.7 罚），别让断言空转
        assert!(
            paths
                .iter()
                .any(|p| p.text == "向打关系吧" && p.penalty > 0.0),
            "paths: {:?}",
            paths
                .iter()
                .map(|p| (p.text.clone(), p.penalty))
                .collect::<Vec<_>>()
        );
        for path in &paths {
            let expected: f64 = path
                .words
                .iter()
                .map(|w| {
                    let freq = dictionary
                        .lookup_exact(
                            &w.syllables
                                .iter()
                                .map(|s| SyllablePattern::complete(s.as_str()))
                                .collect::<Vec<_>>(),
                        )
                        .iter()
                        .find(|m| m.text == w.text)
                        .map(|m| m.frequency)
                        .unwrap_or(0);
                    (f64::from(freq) + 1.0).ln() - log_total + FALLBACK_PENALTY
                })
                .sum();
            assert!(
                ((path.score + path.penalty) - expected).abs() < 1e-9,
                "{}: score {} + penalty {} != {}",
                path.text,
                path.score,
                path.penalty,
                expected
            );
        }
    }

    /// 穷举对拍：小词典上暴力枚举全部词路径的最大分，必须与 convert_paths 的最优一致；
    /// 分歧候选的重算分数若虚高（越过真最优），这里会红。
    #[test]
    fn diverged_candidates_never_beat_the_viterbi_best() {
        let dictionary = Dictionary::parse(
        "那\tna\t90000\n哪\tna\t30000\n一\tyi\t80000\n了\tle\t70000\n一类\tyi lei\t20000\n做\tzuo\t80000\n坐\tzuo\t40000\n了\tle\t70000\n个\tge\t80000\n歪\twai\t3000\n瓜\tgua\t5000\n外观\twai gua\t6000\n做了\tzuo le\t40000\n",
    )
    .unwrap();
        for syllables in [
            vec!["na", "yi", "lei"],
            vec!["zuo", "le", "ge", "wai", "gua"],
        ] {
            let patterns = complete(&syllables);
            let paths = convert_paths(
                &[&dictionary],
                &patterns,
                false,
                16,
                SPAN_CANDIDATES,
                Context::START,
                &NoLanguageModel,
                Personal::NONE,
                |_| 0,
                |_, _| 0.0,
                &mut SpanCache::default(),
            );
            // 穷举：全切分（这里音节数 ≤ 5，词 ≤ 3 音节，直接 DFS）
            let total = fallback_total(&dictionary);
            let best = exhaustive_best(&dictionary, &patterns, total);
            assert!(
                (paths[0].score - best.1).abs() < 1e-9 && paths[0].text == best.0,
                "syllables {syllables:?}: viterbi best {:?}({}) vs exhaustive {:?}({})",
                paths[0].text,
                paths[0].score,
                best.0,
                best.1
            );
        }
    }

    fn fallback_total(dictionary: &Dictionary) -> f64 {
        (dictionary.total_frequency() as f64).max(1.0).ln()
    }

    /// DFS 全部词路径，返回 (文本, 分)。分与 convert_paths 同式：每词 fallback + weight_bonus(0)。
    fn exhaustive_best(
        dictionary: &Dictionary,
        patterns: &[std::vec::Vec<qingjian_dictionary::SyllablePattern<'_>>],
        total: f64,
    ) -> (String, f64) {
        fn walk(
            dictionary: &Dictionary,
            patterns: &[std::vec::Vec<qingjian_dictionary::SyllablePattern<'_>>],
            start: usize,
            total: f64,
            text: &mut String,
            score: &mut f64,
            best: &mut (String, f64),
        ) {
            if start == patterns.len() {
                if *score > best.1 {
                    *best = (text.clone(), *score);
                }
                return;
            }
            for end in start + 1..=patterns.len() {
                let span = &patterns[start..end];
                let pattern: Vec<qingjian_dictionary::SyllablePattern<'_>> = span
                    .iter()
                    .map(|p| *p.first().expect("span 非空"))
                    .collect();
                for hit in dictionary.lookup_exact(&pattern) {
                    if !hit.exact {
                        continue;
                    }
                    let bytes = text.len();
                    text.push_str(hit.text);
                    let step = (f64::from(hit.frequency) + 1.0).ln() - total + FALLBACK_PENALTY;
                    *score += step;
                    walk(dictionary, patterns, end, total, text, score, best);
                    *score -= step;
                    text.truncate(bytes);
                }
            }
        }
        let mut best = (String::new(), f64::NEG_INFINITY);
        let mut text = String::new();
        let mut score = 0.0;
        walk(
            dictionary, patterns, 0, total, &mut text, &mut score, &mut best,
        );
        best
    }

    /// 末位分歧路径：同一结尾词的最优前驱与次优前驱各出一条。只取最优链时，
    /// `waimiandeyu…` 的前 k 条全是 `余下的很大 / 很搭 / 很打` 这类只变末字的近重复，
    /// 排在后面但前缀不同的 `雨下得很大` 进不了重排候选。
    #[test]
    fn paths_include_the_second_best_predecessor_of_the_best_ending() {
        let dictionary = Dictionary::parse(
        "我\two\t900000\n想\txiang\t500000\n我想\two xiang\t600000\n去\tqu\t400000\n区\tqu\t3000\n吃饭\tchi fan\t100000\n",
    )
    .unwrap();
        let patterns = complete(&["wo", "xiang", "qu", "chi", "fan"]);
        let paths = convert_paths(
            &[&dictionary],
            &patterns,
            false,
            3,
            SPAN_CANDIDATES,
            Context::START,
            &NoLanguageModel,
            Personal::NONE,
            |_| 0,
            |_, _| 0.0,
            &mut SpanCache::default(),
        );
        assert_eq!(paths[0].text, "我想去吃饭");
        assert!(
            paths.iter().any(|p| p.text == "我想区吃饭"),
            "paths: {:?}",
            paths.iter().map(|p| p.text.clone()).collect::<Vec<_>>()
        );
    }
}
