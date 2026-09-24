//! 神经重打分：整句转换的前几条路径交给本地模型（[`SentenceScorer`]）再排一次。
//!
//! 打分有两种接法：同步的（[`Engine::with_sentence_scorer`]，查询里当场打，CLI 评测用）和异步的
//! （[`Engine::with_async_sentence_scorer`]，后台线程；壳里用）。两种都经过一张「前文 + 文本 → 神经分」的缓存
//! （[`NeuralCache`]）：同步时缺的分当场补进去，异步时缺的先记下来，壳在用户停顿后调 [`Engine::request_rescoring`]
//! 一次送去后台，[`Engine::poll_rescoring`] 收到结果后再查一次，这时全部路径的分都在缓存里，排序自然换成重排后的。
//! 按键回调永远不等模型：先按词级模型出候选，模型的意见晚几十毫秒到。

mod cache;
mod context;
mod probe;
mod worker;

#[cfg(test)]
mod tests;

use super::*;

pub(crate) use cache::NeuralCache;
pub use context::select_context;
pub(super) use context::split_window;
pub use probe::RerankProbe;
pub(crate) use worker::RescoreWorker;

impl Engine {
    /// 最近一次神经重排的探针快照（没有重排发生是 `None`）。评测用它算:
    /// 重排池 oracle（正确句有没有送进模型）、翻案转化（静态错的被翻对）、翻坏（静态对的被翻错）。
    pub fn rerank_probe(&self) -> Option<RerankProbe> {
        self.rerank_probe.borrow().clone()
    }

    /// 接了重打分器（同步或异步）。
    pub fn has_sentence_scorer(&self) -> bool {
        self.sentence_scorer.is_some()
            || self.rescorer.as_ref().is_some_and(RescoreWorker::is_alive)
    }

    /// 给模型看的双向窗口：总预算 `neural_context` 按 [`split_window`] 自适应切分——
    /// 两侧都有文本对半分（200 字 → 前 100 + 后 100），只有一侧全给那一侧。
    /// 前文按句界结构化装填（[`select_context`]），后文取头部；壳没给前文时退本会话历史（后文照给）。
    pub(super) fn rescoring_window(&self) -> (String, String) {
        if self.neural_context == 0 {
            return (String::new(), String::new());
        }
        let history; // 借的宿主：历史回退分支里要让字符串活到函数尾
        let (before_src, after_src) = match (&self.rescoring_before, &self.rescoring_after) {
            (Some(before), after) => (before.as_str(), after.as_deref().unwrap_or("")),
            (None, after) => {
                history = self.history.recent(self.neural_context * 2);
                (history, after.as_deref().unwrap_or(""))
            }
        };
        let (left_budget, right_budget) = split_window(before_src, after_src, self.neural_context);
        let left = select_context(before_src, left_budget);
        let right: String = after_src.chars().take(right_budget).collect();
        (left, right)
    }

    /// 壳告知应用里光标前的文本（每次查询前给；应用给不出就 `None`，退回本会话历史）。
    pub fn set_rescoring_context(&mut self, before: Option<String>) {
        self.rescoring_before = before;
        self.rescoring_after = None;
    }

    /// 壳告知应用里光标前后的文本，供支持双向上下文的新模型使用。
    pub fn set_rescoring_surrounding(&mut self, before: Option<String>, after: Option<String>) {
        self.rescoring_before = before;
        self.rescoring_after = after;
    }

    /// 把当前存着的应用前后文（重排那次读的）交回去：联想重新要一次上下文时不用再找应用要。
    /// 壳没给过应用文本就是 `None`。
    pub fn rescoring_surrounding(&self) -> Option<SurroundingText> {
        self.rescoring_before
            .as_ref()
            .map(|before| SurroundingText {
                before: before.clone(),
                after: self.rescoring_after.clone().unwrap_or_default(),
            })
    }

    /// 本次重排用的神经修正上限（nat）：用户配置（[`Engine::set_neural_max_adjustment`]）优先，
    /// 其次模型文件自带的建议（[`SentenceScorer::max_adjustment`]），都没有用缺省 [`NEURAL_MAX_ADJUSTMENT`]；
    /// 非有限值与负数一律当没配。
    fn neural_adjustment_cap(&self) -> f64 {
        self.neural_max_adjustment
            .or(self.model_max_adjustment)
            .filter(|cap| cap.is_finite() && *cap >= 0.0)
            .unwrap_or(NEURAL_MAX_ADJUSTMENT)
    }

    /// 把几条整句路径按「路径分 + λ·(神经分 − 静态分)」重排。缓存里缺分的：同步打分器当场补，异步的先记下等壳来取；
    /// 有任何一条没分就不动顺序（半截重排比不重排还糟）。模型分数非有限（NaN / ±inf）的路径跳过调整，
    /// 保持原顺序分量——NaN 进了分数会把排序比较器整个毒掉。
    pub(super) fn rescore_paths(&self, paths: &mut [Conversion]) {
        if paths.len() < 2 || !self.has_sentence_scorer() {
            // 池只有一条:照样记探针(oracle 统计要知道"根本没得选"的句子)
            if paths.len() == 1 {
                *self.rerank_probe.borrow_mut() = Some(RerankProbe {
                    pool: vec![paths[0].text.clone()],
                    ranked: vec![paths[0].text.clone()],
                    top_before: Some(paths[0].text.clone()),
                    top_after: Some(paths[0].text.clone()),
                });
            }
            return;
        }
        let probe_pool: Vec<String> = paths.iter().map(|path| path.text.clone()).collect();
        let top_before = paths.first().map(|path| path.text.clone());
        let (context, after) = self.rescoring_window();
        let cache_context = format!("{context}\0{after}");
        let mut cache = self.neural_cache.borrow_mut();
        cache.ensure_context(&cache_context);
        let mut missing: Vec<String> = Vec::new();
        for path in paths.iter() {
            if cache.get(&path.text).is_none() && !missing.contains(&path.text) {
                missing.push(path.text.clone());
            }
        }
        if !missing.is_empty() {
            match &self.sentence_scorer {
                Some(scorer) => {
                    let texts: Vec<&str> = missing.iter().map(String::as_str).collect();
                    let scores = scorer.score_with_after(&context, &after, &texts);
                    if scores.len() != texts.len() {
                        return;
                    }
                    for (text, score) in texts.iter().zip(scores) {
                        cache.insert(text, score);
                    }
                }
                None => {
                    for text in &missing {
                        cache.want(text);
                    }
                    return;
                }
            }
        }
        let lambda = self.neural_weight;
        let cap = self.neural_adjustment_cap();
        let gate = self.neural_gate;
        // 先把每条路径的调整算好再写回：闸要同时看几条路径的分，边写边比会读到半截状态。
        // 进来的 paths 已按路径分降序排好，下标就是调整前的排名。
        let mut adjusted = Vec::with_capacity(paths.len());
        let mut shifts = Vec::with_capacity(paths.len());
        for path in paths.iter() {
            let neural = cache.get(&path.text).expect("filled above");
            let shift = lambda * (neural - path.static_score);
            // ±inf 会被 clamp 成上限、NaN 穿过 clamp：两者都不该动这条路径的分
            if shift.is_finite() {
                let shift = shift.clamp(-cap, cap);
                adjusted.push(path.score + shift);
                shifts.push(shift);
            } else {
                adjusted.push(path.score);
                shifts.push(0.0);
            }
        }
        // 个人证据保护闸：挑战者要翻掉守成者（老排名靠前）时，神经修正差得追得上守成者的个人证据优势
        //（路径分减静态分——个人 n-gram、用户加分、代价那部分）的 gate 倍，否则压回平手（稳定排序让守成者留在前面）。
        if gate > 0.0 {
            let personal: Vec<f64> = paths
                .iter()
                .map(|path| path.score - path.static_score)
                .collect();
            for keeper in 0..paths.len() {
                for challenger in keeper + 1..paths.len() {
                    if adjusted[challenger] <= adjusted[keeper] {
                        continue;
                    }
                    let neural_gap = shifts[challenger] - shifts[keeper];
                    let personal_gap = personal[keeper] - personal[challenger];
                    if neural_gap < gate * personal_gap {
                        tracing::debug!(
                            keeper = %paths[keeper].text,
                            challenger = %paths[challenger].text,
                            neural_gap,
                            personal_gap,
                            "个人证据闸：神经分差不够，压回平手"
                        );
                        adjusted[challenger] = adjusted[keeper];
                        shifts[challenger] = adjusted[challenger] - paths[challenger].score;
                    }
                }
            }
        }
        for (path, score) in paths.iter_mut().zip(adjusted) {
            path.score = score;
        }
        paths.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        *self.rerank_probe.borrow_mut() = Some(RerankProbe {
            pool: probe_pool,
            ranked: paths.iter().map(|path| path.text.clone()).collect(),
            top_before,
            top_after: paths.first().map(|path| path.text.clone()),
        });
        self.last_rescored.set(true);
    }

    /// 最近一次查询里有整句路径还没拿到神经分：壳该在用户停顿后调 [`Self::request_rescoring`]。
    pub fn rescoring_pending(&self) -> bool {
        self.rescorer.is_some() && self.neural_cache.borrow().has_wanted()
    }

    /// 把攒着的文本送去后台打分。没接异步打分器或没什么要打的返回 `false`。
    /// 相同 (before, after, texts) 的两次请求都真实执行：缓存失效后同文本再来一次正是为了淘汰旧结果。
    pub fn request_rescoring(&mut self) -> bool {
        let Some(worker) = &self.rescorer else {
            return false;
        };
        let mut cache = self.neural_cache.borrow_mut();
        let wanted = cache.take_wanted(RESCORE_BATCH_MAX);
        if wanted.is_empty() {
            return false;
        }
        tracing::debug!(texts = wanted.len(), "神经重打分请求");
        let mut parts = cache.context().split('\0');
        let before = parts.next().unwrap_or_default().to_owned();
        let after = parts.next().unwrap_or_default().to_owned();
        self.rescore_sequence += 1;
        worker.submit(self.rescore_sequence, before, after, wanted);
        true
    }

    /// 收后台打好的分。有新分进了缓存返回 `true`，壳该重新 [`Self::query`] 一次；
    /// 只收序号不小于引擎已发出最大序号的结果——被更新请求顶掉的旧结果即使上下文字符串恰好回到相同的值也丢掉。
    pub fn poll_rescoring(&mut self) -> bool {
        let Some(worker) = &self.rescorer else {
            return false;
        };
        let mut updated = false;
        while let Some(scored) = worker.poll() {
            if scored.sequence < self.rescore_sequence {
                continue;
            }
            let mut cache = self.neural_cache.borrow_mut();
            let expected = cache.context();
            let actual = format!("{}\0{}", scored.before, scored.after);
            if actual != expected || scored.scores.len() != scored.texts.len() {
                continue;
            }
            for (text, score) in scored.texts.iter().zip(scored.scores) {
                cache.insert(text, score);
            }
            updated = true;
        }
        updated
    }

    /// 显式作废攒下的神经分（前文没变也一样）：删空 / 清空缓冲区后，同文本的旧分不再代表当前这段组句的意图，下次重新问模型。
    pub(super) fn forget_neural_cache(&mut self) {
        self.rescore_sequence = self.rescore_sequence.wrapping_add(1);
        self.neural_cache.borrow_mut().clear();
    }
}
