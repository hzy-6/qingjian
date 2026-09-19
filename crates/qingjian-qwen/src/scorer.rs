use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Mutex;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use qingjian_core::sentence::SentenceScorer;

use crate::QwenError;

/// KV 缓存与上下文的 token 上限：前文 64 字加整句 20 字，BPE 后远小于这个数，留足余量。
const CONTEXT_TOKENS: u32 = 1024;

/// CPU 线程数上限：Metal 管矩阵乘，CPU 只做少量收尾，线程多了反而抢核。
const MAX_THREADS: usize = 8;

/// 一次打分最多多少条候选：每条候选占一个序列，超了分批打。
const MAX_TEXTS_PER_CALL: usize = 31;

/// 一批 decode 的 token 预算：llama.cpp 缺省把超过 512 token 的 batch 切成 ubatch，
/// 而混合架构（含 SSM）要求一条序列的 token 不跨 ubatch，这里按预算切批、批内整序列。
const CHUNK_TOKENS: usize = 480;

/// 推理上下文：所有访问都经这把锁串行（llama.cpp 上下文不是线程安全的）。
struct Session {
    context: LlamaContext<'static>,
}

/// 加载好的 Qwen GGUF 模型：给「前文 + 候选」打分。
pub struct QwenScorer {
    model: Box<LlamaModel>,
    session: Mutex<Session>,
}

// SAFETY：llama.cpp 的推理上下文不是线程安全的，但这里所有访问都经 `Mutex` 串行进行，
// 不存在并发进入同一上下文的路径；Core 的 `SentenceScorer` 要求 Send 才能交给后台重排线程。
unsafe impl Send for QwenScorer {}

impl QwenScorer {
    /// 加载 GGUF 模型。`path` 是 llama.cpp 量化文件（如 `Qwen3.5-0.8B-Q8_0.gguf`）。
    pub fn load(path: &Path) -> Result<Self, QwenError> {
        if !path.is_file() {
            return Err(QwenError::NotFound(path.to_owned()));
        }
        let gpu_layers = gpu_layers();
        // SAFETY（泄漏）：llama.cpp 要求后端一个进程只初始化一次，用泄漏换进程级生命周期，退出时一起回收。
        let backend: &'static LlamaBackend = Box::leak(Box::new(
            LlamaBackend::init().map_err(|e| QwenError::Backend(e.to_string()))?,
        ));
        let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);
        let model = Box::new(
            LlamaModel::load_from_file(backend, path, &model_params)
                .map_err(|e| QwenError::Load(e.to_string()))?,
        );
        let threads =
            std::thread::available_parallelism().map_or(4, |n| n.get().min(MAX_THREADS)) as i32;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(CONTEXT_TOKENS))
            .with_n_threads(threads)
            .with_n_threads_batch(threads)
            // 每条候选一个序列；Qwen3.5 的混合注意力（含 SSM 状态）按序列数分配状态
            .with_n_seq_max((MAX_TEXTS_PER_CALL + 1) as u32);
        let context = model
            .new_context(backend, ctx_params)
            .map_err(|e| QwenError::Context(e.to_string()))?;
        // SAFETY：`context` 借用的是 `model`，而 `model` 装在 Box 里、地址在 QwenScorer 存活期间不变，
        // 也不会被移出；把生命周期放宽到 'static 只是绕开「结构体存自己字段的借用」这个表达限制。
        let context: LlamaContext<'static> = unsafe { std::mem::transmute(context) };
        tracing::info!(
            source = %path.display(),
            params = model.n_params(),
            gpu_layers,
            threads,
            "Qwen 模型已加载"
        );
        Ok(Self {
            model,
            session: Mutex::new(Session { context }),
        })
    }

    /// 前文 token：模型要 BOS 就带上；空前文给一个文档起始 token 当条件（Qwen 没设 decoder 起始 token，
    /// 拿到 -1 就退到 BOS / EOS），让候选首 token 也有分布。三者都无效才是真空前文（首 token 不打分）。
    fn prefix_tokens(&self, context: &str) -> Vec<LlamaToken> {
        let tokens = self
            .model
            .str_to_token(context, AddBos::Always)
            .unwrap_or_default();
        if !tokens.is_empty() {
            return tokens;
        }
        [
            self.model.decode_start_token(),
            self.model.token_bos(),
            self.model.token_eos(),
        ]
        .into_iter()
        .find(|token| token.0 >= 0)
        .map(|token| vec![token])
        .unwrap_or_default()
    }

    /// 每个候选接在 `context` 后面的 `log P(候选 | 前文)`，按 BPE token 累加。
    /// 每条候选拼上前文各自成一个序列，一次 decode 打一批，打完清空 KV
    /// （Qwen3.5 是注意力 + SSM 的混合架构：`seq_cp` 会整块拷贝 KV，循环状态也回卷不了中间位置，
    /// 都不如「整序列重算」来得简单可靠；前文几十个 token 在 Metal 上重算很便宜）。
    fn score_inner(&self, context: &str, texts: &[&str]) -> Result<Vec<f64>, QwenError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let prefix = self.prefix_tokens(context);
        let tails: Vec<Vec<LlamaToken>> = texts
            .iter()
            .map(|text| {
                self.model
                    .str_to_token(text, AddBos::Never)
                    .map_err(|e| QwenError::Tokenize(e.to_string()))
            })
            .collect::<Result<_, _>>()?;
        let longest = tails.iter().map(Vec::len).max().unwrap_or(0);
        if prefix.len() + longest >= CONTEXT_TOKENS as usize {
            return Err(QwenError::TooLong {
                prefix: prefix.len(),
                longest,
            });
        }
        let mut session = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut scores: Vec<f64> = vec![0.0; tails.len()];
        for chunk in chunks(&tails, prefix.len()) {
            let scored = decode_chunk(&mut session.context, &prefix, &tails, &chunk)?;
            for (slot, score) in scored {
                scores[slot] = score;
            }
        }
        Ok(scores)
    }
}

impl SentenceScorer for QwenScorer {
    fn score(&self, context: &str, texts: &[&str]) -> Vec<f64> {
        self.score_inner(context, texts).unwrap_or_else(|error| {
            tracing::warn!(%error, "Qwen 打分失败，这条当没有神经分");
            Vec::new()
        })
    }

    /// 修正建议:30 nat——不是训练侧给的,是 Qwen 接管时在 139 句盲评上扫出的缺省(BPE 按 token 累加的神经分
    /// 比字级的分差大,12 的旧缺省会削顶:88.5 → 89.2;40 与 30 同分,留余量给长分歧链)。经 `max_adjustment`
    /// 机制带给 Core,壳不传参数即用这个值(见 docs/notes/qwen-rescoring.md)。
    fn max_adjustment(&self) -> Option<f64> {
        Some(30.0)
    }

    /// 双向上下文补分（与 `qingjian-neural::CharScorer::score_with_after` 同一配方）：
    /// 正向分为主，右文项比较「候选 + 右文」与「右文」在同一前文下的增量，按 0.25 折进总分。
    /// 后文为空走快速路径，不多做推理；右文项打不出（模型出错）就只给正向分。
    fn score_with_after(&self, before: &str, after: &str, texts: &[&str]) -> Vec<f64> {
        let Some(forward) = self.score_inner(before, texts).ok() else {
            return Vec::new();
        };
        if after.is_empty() || texts.is_empty() {
            return forward;
        }
        let Some(baseline) = self
            .score_inner(before, &[after])
            .ok()
            .and_then(|mut v| v.pop())
        else {
            return forward;
        };
        let joined: Vec<String> = texts.iter().map(|text| format!("{text}{after}")).collect();
        let joined_refs: Vec<&str> = joined.iter().map(String::as_str).collect();
        match self.score_inner(before, &joined_refs) {
            Ok(joint) => forward
                .into_iter()
                .zip(joint)
                .map(|(base, combined)| base + 0.25 * (combined - baseline))
                .collect(),
            Err(_) => forward,
        }
    }
}

/// 按序列条数与 token 预算切块：返回每块的候选下标（`tails` 里空的候选不进块、分数记 0）。
fn chunks(tails: &[Vec<LlamaToken>], prefix_len: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut current = Vec::new();
    let mut tokens = 0;
    for (index, tail) in tails.iter().enumerate() {
        let len = tail.len();
        if len == 0 {
            continue;
        }
        let seq_len = prefix_len + len;
        let full = current.len() >= MAX_TEXTS_PER_CALL || tokens + seq_len > CHUNK_TOKENS;
        if full && !current.is_empty() {
            out.push(std::mem::take(&mut current));
            tokens = 0;
        }
        current.push(index);
        tokens += seq_len;
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// 一块候选一次 decode：每条候选拼上前文各自成一序列，首 token 到倒数第二个 token 都取分布，
/// 按 token 累加 log 概率。返回（候选下标，分数）；打完清空 KV，出错也先清再冒泡。
fn decode_chunk(
    context: &mut LlamaContext<'_>,
    prefix: &[LlamaToken],
    tails: &[Vec<LlamaToken>],
    slots: &[usize],
) -> Result<Vec<(usize, f64)>, QwenError> {
    let mut batch = LlamaBatch::new(
        slots
            .iter()
            .map(|&slot| prefix.len() + tails[slot].len())
            .sum(),
        // 第二参数是每个 token 的 seq id 数组容量（n_seq_max），不是 embd：每个 token 只写一个序列 id，1 就够
        1,
    );
    // 每条序列的起始 batch 下标：读 logits 时按全局下标取
    let mut offsets = Vec::with_capacity(slots.len());
    let mut offset = 0;
    for (nth, &slot) in slots.iter().enumerate() {
        let seq = (nth + 1) as i32;
        let seq_tokens: Vec<&LlamaToken> = prefix.iter().chain(tails[slot].iter()).collect();
        offsets.push(offset);
        for (i, token) in seq_tokens.iter().enumerate() {
            // 前文最后一个 token 给候选首 token 出分布；候选最后一个 token 不用出自己的分布
            let wants_logits = i + 1 < seq_tokens.len() && i + 1 >= prefix.len();
            batch
                .add(**token, i as i32, &[seq], wants_logits)
                .map_err(|e| QwenError::Decode(e.to_string()))?;
        }
        offset += seq_tokens.len();
    }
    let decoded = context
        .decode(&mut batch)
        .map_err(|e| QwenError::Decode(e.to_string()));
    context.clear_kv_cache();
    decoded?;
    let mut out = Vec::with_capacity(slots.len());
    for (nth, &slot) in slots.iter().enumerate() {
        let tail = &tails[slot];
        let mut sum = 0.0f32;
        for (i, token) in tail.iter().enumerate() {
            // 真空前文（连文档起始 token 都没有）时首 token 没有条件分布，跳过
            if prefix.len() + i == 0 {
                continue;
            }
            // 候选第 i 个 token 的分布在前文最后一个（i=0）或它前一个 token 的位置上
            let batch_index = (offsets[nth] + prefix.len() + i - 1) as i32;
            let logits = context.get_logits_ith(batch_index);
            // 只需要一个 token 的值：单遍求 logsumexp,不把 ~150k 的整个分布物化成 Vec
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = logits.iter().map(|&v| (v - max).exp()).sum::<f32>().ln();
            let logit = u32::try_from(token.0)
                .ok()
                .and_then(|id| logits.get(id as usize))
                .copied()
                .unwrap_or(f32::NEG_INFINITY);
            sum += logit - max - lse;
        }
        out.push((slot, f64::from(sum)));
    }
    Ok(out)
}

/// GPU 卸载层数：缺省全部上 Metal，`QINGJIAN_QWEN_GPU_LAYERS` 可覆盖（0 为纯 CPU）。
fn gpu_layers() -> u32 {
    std::env::var("QINGJIAN_QWEN_GPU_LAYERS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(99)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    /// 本机下载的 Qwen GGUF（`data/model/`，gitignore）；没有就跳过。
    fn shipped_gguf() -> Option<std::path::PathBuf> {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/model/Qwen3.5-0.8B-Q8_0.gguf");
        path.is_file().then_some(path)
    }

    /// 通顺的候选分数要明显高于同音错的，重复调用结果一致，分批与超量也正确。
    #[test]
    fn prefers_the_fluent_candidate_and_batches_correctly() {
        let Some(path) = shipped_gguf() else {
            eprintln!("没有 Qwen GGUF，跳过");
            return;
        };
        let scorer = QwenScorer::load(&path).unwrap();
        // 模型自带的修正建议要带给 Core（壳不传参数时的默认上限）
        assert_eq!(scorer.max_adjustment(), Some(30.0));
        let scores = scorer
            .score_inner("我今天想去", &["上海", "伤害", "吃饭"])
            .unwrap();
        assert_eq!(scores.len(), 3);
        assert!(
            scores.iter().all(|s| s.is_finite() && *s < 0.0),
            "{scores:?}"
        );
        assert!(scores[0] > scores[1] + 2.0, "{scores:?}");
        // 单独打与批量打一致（多序列与单序列走不同的 Metal 归约核，长句的浮点差到千分位）
        let alone = scorer.score_inner("我今天想去", &["伤害"]).unwrap();
        assert!(
            (alone[0] - scores[1]).abs() < 5e-3,
            "{alone:?} vs {scores:?}"
        );
        // 空前文也能打分
        let bare = scorer.score_inner("", &["上海"]).unwrap();
        assert!(bare[0] < 0.0, "{bare:?}");
        // 一次塞超过序列容量 / token 预算的候选也能分批打完，且与单独打一致
        let many: Vec<String> = (0..40).map(|i| format!("测试句子第{i}条")).collect();
        let refs: Vec<&str> = many.iter().map(String::as_str).collect();
        let batched = scorer.score_inner("前文", &refs).unwrap();
        assert_eq!(batched.len(), 40);
        let single = scorer.score_inner("前文", &[refs[7]]).unwrap();
        assert!(
            (batched[7] - single[0]).abs() < 5e-3,
            "{} vs {}",
            batched[7],
            single[0]
        );
        // 空候选记 0 分
        let empty = scorer.score_inner("前文", &["", "上海"]).unwrap();
        assert_eq!(empty[0], 0.0);
        assert!(empty[1] < 0.0);
        // 双向：空后文走快速路径（与 score 等值，多序列与单序列的 Metal 归约有 1e-3 量级浮点差）
        use qingjian_core::sentence::SentenceScorer as _;
        let plain = scorer.score_with_after("我今天想去", "", &["上海"]);
        assert_eq!(plain.len(), 1);
        assert!(
            (plain[0] - scores[0]).abs() < 5e-3,
            "{plain:?} vs {scores:?}"
        );
        let both = scorer.score_with_after("我今天想去", "吃饭", &["上海", "商量"]);
        assert_eq!(both.len(), 2);
        assert!(both.iter().all(|s| s.is_finite()), "{both:?}");
        assert!(
            (both[0] - scores[0]).abs() > 1e-6,
            "右文项应该改分：{both:?} vs {scores:?}"
        );
    }
}
