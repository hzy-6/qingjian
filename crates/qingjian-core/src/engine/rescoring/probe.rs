/// 一次神经重排的探针快照:重排池(按静态排名进池的顺序)与重排前后的首选。
/// 评测用它算 oracle 指标(正确句有没有送进模型)与翻案方向(转化 / 翻坏),生产路径不读。
#[derive(Debug, Clone, PartialEq)]
pub struct RerankProbe {
    /// 重排池里的路径文本,按进入时的静态排名排。
    pub pool: Vec<String>,

    /// 重排前(静态)排第一的路径;池只有一条或没接打分器时与重排后一致。
    pub top_before: Option<String>,

    /// 重排后排第一的路径。
    pub top_after: Option<String>,
}
