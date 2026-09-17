//! 整段拼写纠错：找一两处编辑的纠正并按个人敲错表打折。

use super::*;

/// 双错联合纠错的预算：第二处编辑的「生成 + 切分检查」最多做这么多次。
/// 单错候选几十个、每个的高频第二变体百十个，放开了是几万次检查；预算之内找不到就认输。
const SECOND_EDIT_BUDGET: usize = 6144;

/// 双错候选最多做几次整句转换：过了切分检查的候选可能仍有几百个，全部转换会让每次查询跟着变贵。
const DOUBLE_SCORE_BUDGET: usize = 192;

/// 一个种子最多做几次整句转换：免得排在前面的种子把转换预算烧光，后面的种子连机会都没有。
const SEED_CONVERSION_CAP: usize = 12;

/// 双错搜索的种子：一个单错候选加上「第二处错误」的锚点。
/// 候选的整句得分不进种子——单错的选择在生成种子的同一次遍历里按变体原顺序定好（见 [`Engine::seeds`]）。
struct Seed {
    /// 第二处编辑优先试的位置（纠正后串的字母下标）：整句转出占位音节的种子（那个音节在词库里
    /// 一个词都凑不出）取占位洞的起点，其余取原串的断点。词库里单字缺词（`nen` 只有 嫩滑 没有 嫩）不该把路堵死。
    anchor: usize,

    correction: Correction,
}

/// 双错种子的优先级：换位与相邻键替换最像真实的敲错（0），多敲 / 少敲其次（1），远距替换垫底（2）。
fn seed_rank(candidate: &Correction) -> usize {
    match candidate.edit {
        correction::Edit::Transpose { .. } => 0,
        correction::Edit::Substitute { index, .. } => {
            let from = candidate.original.as_bytes().get(index).copied();
            let to = candidate.corrected.as_bytes().get(index).copied();
            match (from, to) {
                (Some(from), Some(to)) if typo::adjacent(from as char, to as char) => 0,
                _ => 2,
            }
        }
        correction::Edit::Delete { .. } | correction::Edit::Insert { .. } => 1,
    }
}

/// 整句转换里第一个占位词在切分里的字母区间：词按顺序吃音节，数一数它前面吃掉几个就是了。
fn hole_span(conversion: &Conversion, segmentation: &Segmentation) -> Option<(usize, usize)> {
    let mut syllables = 0;
    for word in &conversion.words {
        if word.placeholder {
            let start: usize = segmentation
                .syllables
                .iter()
                .take(syllables)
                .map(|s| s.text.len())
                .sum();
            let end: usize = start
                + segmentation
                    .syllables
                    .iter()
                    .skip(syllables)
                    .take(word.syllables.len())
                    .map(|s| s.text.len())
                    .sum::<usize>();
            return Some((start, end));
        }
        syllables += word.syllables.len();
    }
    None
}

impl Engine {
    /// 这段作用域生效的拼写纠正（带缓存）：拼音不像话、用户没对它回车原样上屏过、
    /// 且一处编辑后能凑出至少一个两音节词时，取整句转换得分最高的那个纠正；
    /// 单错压不过原样（或根本没有像样的单错）时，再试两处编辑的联合纠错。
    pub(super) fn active_correction(&self, scope: &str) -> Option<Correction> {
        // 双拼敲错一个键换掉的是整个声母 / 韵母，全拼那套「一处编辑」的纠错模型不适用
        if self.shuangpin.is_some() {
            return None;
        }
        if let Some((cached_scope, cached)) = self.correction_cache.borrow().as_ref()
            && cached_scope == scope
        {
            return cached.clone();
        }
        let found = self.find_correction(scope);
        *self.correction_cache.borrow_mut() = Some((scope.to_owned(), found.clone()));
        found
    }

    pub(super) fn find_correction(&self, scope: &str) -> Option<Correction> {
        if !correction::eligible(scope) || self.learner.raw_count(scope) > 0 {
            return None;
        }
        // 整段就是个英文词（hello）：用户多半在打英文，别把它「纠」成 喝了哦
        if self
            .english
            .as_ref()
            .is_some_and(|english| english.get(scope).is_some())
        {
            return None;
        }
        let (segmentations, tail) = segment_longest_prefix(scope).ok()?;
        // 不像话的拼音试全部一处编辑；末尾单字母的只试相邻换位（`mingtain` → `mingtian`），其余合法拼音不碰
        let candidates = if correction::unlikely_pinyin(segmentations.first(), tail) {
            correction::candidates(scope)
        } else if correction::trailing_single_letter(segmentations.first()) {
            correction::transposition_candidates(scope)
        } else {
            return None;
        };
        // 噪声信道：原串按原样能转出的整句得分 vs 纠正后的整句得分扣掉编辑的代价，后者高才纠。
        // 原串切不干净（有尾巴）就没有原样得分
        let raw_score = if tail.is_empty() {
            segmentations
                .first()
                .and_then(|best| self.convert_sentence(&best.patterns(), true))
                .filter(|conversion| !conversion.has_placeholder())
                .map(|conversion| conversion.score)
        } else {
            None
        };
        // 断点：尾巴的起点，或第一个残缺音节的起点。第二处敲错多半就在它附近。
        let consumed = tail.is_empty().then(|| {
            segmentations
                .first()
                .map(|best| {
                    best.syllables
                        .iter()
                        .take(best.syllables.len().saturating_sub(1))
                        .take_while(|s| s.complete)
                        .map(|s| s.text.len())
                        .sum::<usize>()
                })
                .unwrap_or_default()
        });
        let break_at = consumed.map_or(scope.len() - tail.len(), |consumed| {
            if consumed == 0 { scope.len() } else { consumed }
        });
        let (seeds, best_single) = self.seeds(scope, &candidates, break_at);
        // 修好第一处之后整句就说得通的：维持只纠一处的现状，不花双错搜索的工夫
        if let Some((score, found)) = best_single.as_ref()
            && raw_score.is_some_and(|raw| *score > raw)
        {
            tracing::debug!(original = %found.original, corrected = %found.corrected, score, raw = raw_score, "拼写纠正");
            return Some(found.clone());
        }
        // 双错的采用门槛：压过原样（没有原样分就拿最好的单错当参照）再高出一个安全余量，
        // 两处都改的面更大，改错了比不纠更烦。没有参照就不比了——原样说不通、单错也修不出
        // 像样读法的串（`kaiv` 留着尾巴等下一键），两处编辑凑出来的词多半是瞎猜。
        if let Some(reference) = raw_score.or(best_single.as_ref().map(|(score, _)| *score)) {
            let floor = reference + self.typo_costs.double_error_margin;
            if let Some((score, found)) = self.best_double(&seeds, floor) {
                tracing::debug!(
                    original = %found.original,
                    corrected = %found.corrected,
                    second_edit = true,
                    score,
                    raw = raw_score,
                    "两处编辑的拼写纠正"
                );
                return Some(found);
            }
        }
        match best_single {
            Some((score, found)) => {
                if raw_score.is_some_and(|raw| score <= raw) {
                    tracing::debug!(
                        original = %found.original,
                        corrected = %found.corrected,
                        score,
                        raw = raw_score,
                        "原样已经说得通，不纠"
                    );
                    return None;
                }
                tracing::debug!(original = %found.original, corrected = %found.corrected, score, raw = raw_score, "拼写纠正");
                Some(found)
            }
            None => None,
        }
    }

    /// 单错候选整理成双错搜索的种子：按敲错类型的可信度排序（换位 / 相邻键替换最先），
    /// 同级保持变体原顺序。顺带算好每个种子的整句得分与锚点。
    /// 返回的第二个值是**变体原顺序**里得分最高的单错（严格大于才更新，平局保留靠前的——
    /// 与只纠一处时的选择完全一致；种子重排只服务双错搜索，不许改变单错的结果）。
    fn seeds(
        &self,
        scope: &str,
        candidates: &[Correction],
        break_at: usize,
    ) -> (Vec<Seed>, Option<(f64, Correction)>) {
        let mut seeds: Vec<Seed> = Vec::new();
        let mut best_single: Option<(f64, Correction)> = None;
        for candidate in candidates {
            // 「删掉刚敲的最后一个字母」不算纠正：用户可能还没敲完，尾巴留着等下一键
            if matches!(candidate.edit, correction::Edit::Delete { index, .. } if index + 1 == scope.len())
            {
                continue;
            }
            // 纠正后的拼音上不再猜音节级敲错：变体本来就是一处编辑之外的读法，再叠一层既慢又几乎不会赢
            let conversion = self.convert_sentence(&candidate.segmentation.patterns(), false);
            let anchor = match conversion {
                Some(conversion) if !conversion.has_placeholder() => {
                    let accepted = candidate
                        .typo_pair(candidate.corrected.len())
                        .map_or(0, |(typed, intended)| {
                            self.learner.typo_count(&typed, &intended)
                        });
                    let score = conversion.score - self.typo_costs.correction_cost(accepted);
                    if best_single.as_ref().is_none_or(|(best, _)| score > *best) {
                        best_single = Some((score, candidate.clone()));
                    }
                    break_at
                }
                // 带占位的种子：占位的位置多半就是第二处敲错，把它当锚点
                conversion => conversion
                    .as_ref()
                    .and_then(|conversion| hole_span(conversion, &candidate.segmentation))
                    .map_or(break_at, |(start, _)| start),
            };
            seeds.push(Seed {
                anchor,
                correction: candidate.clone(),
            });
        }
        let mut with_rank = seeds
            .into_iter()
            .map(|seed| (seed_rank(&seed.correction), seed))
            .collect::<Vec<_>>();
        // 稳定排序：同级之内保持变体原顺序（换位最先、下标从前往后）
        with_rank.sort_by_cached_key(|(rank, _)| *rank);
        (
            with_rank.into_iter().map(|(_, seed)| seed).collect(),
            best_single,
        )
    }

    /// 双错联合纠错：把每个单错候选当种子再叠一处高频编辑（[`correction::second_variants`]），
    /// 离锚点近的第二处编辑先试，整句得分（扣两次纠错代价与第二处的额外惩罚）第一个超过
    /// `floor` 的就采用——门槛已经足够苛刻，垃圾凑句差着十几 nat，不必再挑最优。
    /// 只在单错救不回来时才调；预算把检查与整句转换的次数钉死，最坏情形也不会拖慢查询。
    /// 种子必须自己能完整切分：两处错各自拆开都凑不出合法切分的极端组合不在此列，宁缺毋滥。
    fn best_double(&self, seeds: &[Seed], floor: f64) -> Option<(f64, Correction)> {
        let mut checks = SECOND_EDIT_BUDGET;
        let mut conversions = DOUBLE_SCORE_BUDGET;
        for seed in seeds {
            if checks == 0 || conversions == 0 {
                break;
            }
            let first = &seed.correction;
            let mut variants = correction::second_variants(&first.corrected);
            // 离锚点越近的第二处编辑越先试：锚点是占位洞或原串的断点，第二处敲错多半就在旁边
            variants.sort_by_cached_key(|(edit, _)| {
                (edit.position().abs_diff(seed.anchor), edit.position())
            });
            let mut quota = SEED_CONVERSION_CAP;
            for (edit, corrected) in variants {
                if checks == 0 || conversions == 0 || quota == 0 {
                    break;
                }
                checks -= 1;
                // 第二处也别是「删掉刚敲的最后一个字母」：用户可能还没敲完
                if matches!(edit, correction::Edit::Delete { index, .. } if index + 1 == first.corrected.len())
                {
                    continue;
                }
                // 先用便宜的布尔检查过滤，过了的少数再做真正的切分（同 `candidates_from`）
                if !parser::is_fully_segmentable(&corrected) {
                    continue;
                }
                let Some(segmentation) = correction::complete_segmentation(&corrected) else {
                    continue;
                };
                let Some(conversion) = self.convert_sentence(&segmentation.patterns(), false)
                else {
                    continue;
                };
                // 转一次就记一次账（带占位音节的也不例外：Viterbi 照跑全价），
                // 否则词库凑不出单字的音节能把预算绕过去，192 的上限就成了空话
                conversions -= 1;
                quota -= 1;
                if conversion.has_placeholder() {
                    continue;
                }
                let candidate = Correction {
                    original: first.original.clone(),
                    corrected,
                    edit: first.edit.clone(),
                    second: Some(edit),
                    segmentation,
                };
                let pairs = candidate.typo_pairs(candidate.corrected.len());
                let accepted = |index: usize| {
                    pairs.get(index).map_or(0, |(typed, intended)| {
                        self.learner.typo_count(typed, intended)
                    })
                };
                let score = conversion.score
                    - self
                        .typo_costs
                        .double_correction_cost(accepted(0), accepted(1));
                if score > floor {
                    tracing::debug!(corrected = %candidate.corrected, score, floor, "双错候选过关");
                    return Some((score, candidate));
                }
            }
        }
        None
    }
}
