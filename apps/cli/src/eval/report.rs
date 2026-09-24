use std::fmt;
use std::time::Duration;

/// 整句评测报告。
#[derive(Debug, Default)]
pub struct Report {
    /// 抽出来的句子数（去重后）。
    pub extracted: usize,

    /// 有字查不到读音、没法转拼音的句子数。
    pub untranscribable: usize,

    /// 评了的句子数。
    pub total: usize,

    /// 拼音现在切不动的句子数。
    pub unparsable: usize,

    /// 首选就是原句。
    pub top1: usize,

    /// 整句候选（第一个盖住全部拼音的候选）就是原句。
    pub sentence_hit: usize,

    /// 整句候选与原句逐字比对：对上的字数 / 总字数。
    pub chars_correct: usize,
    pub chars_total: usize,

    /// 查询耗时之和与最大值。
    pub query_time: Duration,
    pub slowest_query: Duration,

    /// 没命中首选的例子。
    pub misses: Vec<String>,

    /// 有重排探针的句子数(分母;没接打分器时为 0)。
    pub probed: usize,

    /// 重排池里含原句的句子数(oracle:正确句有没有送进模型)。
    pub oracle_hit: usize,

    /// 池内原句名次之和与命中数(算平均名次)。
    pub oracle_rank_sum: usize,

    /// 静态首选不是原句、重排后变成原句的句子数(翻案转化)。
    pub converted: usize,

    /// 静态首选是原句、重排后不再是原句的句子数(翻坏)。
    pub harmful_flips: usize,

    /// 静态首选就是原句的句子数(翻坏的分母)。
    pub static_top1: usize,

    /// 纠错探针：发出本地联想请求并拿到结果的句子数（分母，没开本地联想时为 0）。
    pub correction_probed: usize,

    /// 纠错槽里含原句的句子数。
    pub correction_hit: usize,

    /// 首选不是原句、但纠错槽把它救回来的句子数。
    pub correction_rescued: usize,

    /// 首选本来就是原句、纠错槽却有别的词的句子数（噪声 / 误纠）。
    pub correction_noise: usize,

    /// 等本地联想超时的句子数。
    pub correction_timeout: usize,
}

impl Report {
    pub fn evaluated(&self) -> usize {
        self.total - self.unparsable
    }
}

fn percent(part: usize, whole: usize) -> String {
    if whole == 0 {
        "-".to_owned()
    } else {
        format!("{:.1}%", part as f64 * 100.0 / whole as f64)
    }
}

impl Report {
    /// 池内原句平均名次(1 起;只算进了池的)。
    pub fn oracle_avg_rank(&self) -> f64 {
        if self.oracle_hit == 0 {
            0.0
        } else {
            self.oracle_rank_sum as f64 / self.oracle_hit as f64
        }
    }
}
impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "整句评测（冷启动，不学习，不写文件）")?;
        let evaluated = self.evaluated();
        writeln!(
            f,
            "句子 {:>5} 条  首选 {:>6}  整句候选 {:>6}  字准确率 {:>6}",
            self.total,
            percent(self.top1, evaluated),
            percent(self.sentence_hit, evaluated),
            percent(self.chars_correct, self.chars_total),
        )?;
        if evaluated > 0 {
            writeln!(
                f,
                "查询平均 {:.1} ms，最慢 {:.1} ms",
                self.query_time.as_secs_f64() * 1000.0 / evaluated as f64,
                self.slowest_query.as_secs_f64() * 1000.0,
            )?;
        }
        if self.probed > 0 {
            writeln!(
                f,
                "重排池 oracle {:>6}（池内平均名次 {:.2}）  翻案转化 {:>6}  翻坏 {:>6}（静态首选对 {}/{}）",
                percent(self.oracle_hit, self.probed),
                self.oracle_avg_rank(),
                percent(self.converted, self.probed.saturating_sub(self.static_top1)),
                percent(self.harmful_flips, self.static_top1),
                self.static_top1,
                self.probed,
            )?;
        }
        if self.unparsable > 0 {
            writeln!(f, "其中 {} 条拼音切不动", self.unparsable)?;
        }
        if self.correction_probed > 0 {
            writeln!(
                f,
                "本地联想/纠错 {:>6}（{} / {}）  救回 {:>6}  首选已对却有纠错 {:>6}  超时 {}",
                percent(self.correction_hit, self.correction_probed),
                self.correction_hit,
                self.correction_probed,
                percent(
                    self.correction_rescued,
                    self.correction_probed - self.static_top1.min(self.correction_probed)
                ),
                percent(
                    self.correction_noise,
                    self.static_top1.min(self.correction_probed)
                ),
                self.correction_timeout,
            )?;
        }
        if self.untranscribable > 0 {
            writeln!(
                f,
                "抽出 {} 条，{} 条有字查不到读音，跳过",
                self.extracted, self.untranscribable
            )?;
        }
        if !self.misses.is_empty() {
            writeln!(f, "\n没命中首选的例子：")?;
            for miss in &self.misses {
                writeln!(f, "  {miss}")?;
            }
        }
        Ok(())
    }
}
