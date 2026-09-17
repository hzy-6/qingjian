//! 私密输入：壳判定应用声明「别记」的输入范围。
//! Core 在这里停止学习、记录输入日志和发出云端请求；排序仍用已有的个人数据。

use super::Engine;

impl Engine {
    /// 进入 / 离开私密输入。壳在焦点落到私密输入框（或离开它）时调；跨会话切换焦点时按各会话的状态重设。
    /// 此方法只设置写入开关，不清组句；已确认的隐私能力边界应先调用 [`Self::discard_input`]，
    /// 避免私密缓存被普通输入继续使用。恢复另一个独立会话时不需要丢弃该会话的输入。
    pub fn set_private(&mut self, private: bool) {
        if self.private == private {
            return;
        }
        self.private = private;
        self.learner.set_private(private);
        self.logger.set_muted(private);
        if private {
            // 在飞的云结果不能再显示，前文也不能留
            self.prediction_sequence += 1;
            self.rescoring_before = None;
            self.rescoring_after = None;
            self.forget_neural_cache();
        }
    }

    pub fn is_private(&self) -> bool {
        self.private
    }
}
