//! 给神经重排模型选上下文:按句界结构化截取,不做机械的「最后 N 字」。

/// 自适应窗口：总预算 `budget`，按两侧有无文本动态分——
/// 只有前文全给前文；只有后文全给后文；两侧都有各拿一半，短的一侧用不完的让给另一侧。
/// 返回 (前文字符数, 后文字符数)；前文取尾部（最近的话），后文取头部，由调用方裁。
pub fn split_window(before: &str, after: &str, budget: usize) -> (usize, usize) {
    let before_chars = before.chars().count();
    let after_chars = after.chars().count();
    if after_chars == 0 {
        return (before_chars.min(budget), 0);
    }
    if before_chars == 0 {
        return (0, after_chars.min(budget));
    }
    let left = before_chars.min(budget / 2);
    let right = after_chars.min(budget - left);
    // 右侧没用完的预算回让给左侧(前文是主要证据)
    let left = left + (before_chars - left).min(budget - left - right);
    (left, right)
}

/// 句界字符:这些之后开新句。用于把光标前文切成句段。
const SENTENCE_BOUNDARIES: [char; 8] = ['。', '！', '？', '!', '?', ';', '；', '\n'];

/// 一个句段里连续非中日韩字符超过这么长,视为代码 / URL / 日志串,不进模型上下文。
const JUNK_ASCII_RUN: usize = 24;

/// 按句界选至多 `budget` 个字符的模型上下文:
/// 从最近的句段往前装,装满为止;纯代码 / URL / 日志的长串(没有汉字、ASCII 连跑超限)整段跳过,
/// 别让它挤掉自然语言;单句超预算时取该句的尾部(与旧截取一致)。
pub fn select_context(before: &str, budget: usize) -> String {
    if budget == 0 || before.is_empty() {
        return String::new();
    }
    if before.chars().count() <= budget {
        return before.to_owned();
    }
    // 按句界切段(边界字符留在段尾),段序即文序
    let mut segments: Vec<String> = Vec::new();
    let mut current = String::new();
    for c in before.chars() {
        current.push(c);
        if SENTENCE_BOUNDARIES.contains(&c) {
            segments.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        segments.push(current);
    }
    // 从最后一段往前装;非末段的「垃圾段」(无汉字且 ASCII 连跑超限)跳过
    let mut kept: Vec<&str> = Vec::new();
    let mut used = 0usize;
    for (index, segment) in segments.iter().enumerate().rev() {
        let len = segment.chars().count();
        let is_last = index + 1 == segments.len();
        if !is_last && is_junk(segment) {
            continue;
        }
        if used + len > budget {
            // 这段装不下:是唯一一段就取尾部;否则到此为止,前面不再装
            if kept.is_empty() {
                let tail: String = segment.chars().skip(len.saturating_sub(budget)).collect();
                return tail;
            }
            break;
        }
        used += len;
        kept.push(segment);
    }
    kept.iter().rev().map(|s| s.to_string()).collect::<String>()
}

/// 垃圾段:没有汉字,且最长 ASCII 连跑超过阈值(代码 / URL / 日志 / 纯数字串)。
fn is_junk(segment: &str) -> bool {
    let has_han = segment.chars().any(is_han_char);
    if has_han {
        return false;
    }
    let mut run = 0usize;
    let mut longest = 0usize;
    for c in segment.chars() {
        if c.is_ascii() {
            run += 1;
            longest = longest.max(run);
        } else {
            run = 0;
        }
    }
    longest > JUNK_ASCII_RUN
}

/// 与 Core `sentence::text_segment::is_han` 同一套范围(这里避免跨模块依赖复制一份判定)。
fn is_han_char(c: char) -> bool {
    matches!(c as u32, 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x323AF)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_short_inputs_pass_through() {
        assert_eq!(select_context("", 64), "");
        assert_eq!(select_context("你好", 64), "你好");
        assert_eq!(select_context("你好", 0), "");
    }

    #[test]
    fn split_window_adapts_to_what_each_side_has() {
        // 两侧都有：对半
        assert_eq!(
            split_window(&"前".repeat(150), &"后".repeat(150), 200),
            (100, 100)
        );
        // 只有一侧：全给那一侧，另一侧为零
        assert_eq!(split_window(&"前".repeat(150), "", 200), (150, 0));
        assert_eq!(split_window(&"前".repeat(300), "", 200), (200, 0));
        assert_eq!(split_window("", &"后".repeat(90), 200), (0, 90));
        // 短的一侧用不完，剩下的让给对侧
        assert_eq!(
            split_window(&"前".repeat(30), &"后".repeat(300), 200),
            (30, 170)
        );
        assert_eq!(
            split_window(&"前".repeat(300), &"后".repeat(10), 200),
            (190, 10)
        );
        // 两侧都空
        assert_eq!(split_window("", "", 200), (0, 0));
    }

    #[test]
    fn a_single_long_sentence_takes_its_tail() {
        let text = "这是一句非常长的话".repeat(20);
        let got = select_context(&text, 10);
        assert_eq!(got.chars().count(), 10);
        assert!(text.ends_with(&got));
    }

    #[test]
    fn fills_from_sentence_boundaries_in_order() {
        let text = "第一句话。第二句话。第三句话。";
        // 每段 5 字:预算 12 装不下三段(15),最前的段装不下就停,保住最近两句(10 字)
        let got = select_context(text, 12);
        assert_eq!(got, "第二句话。第三句话。");
        // 预算正好等于全部(15)时三段都在
        assert_eq!(select_context(text, 15), text);
    }

    #[test]
    fn junk_segments_are_skipped_not_packed() {
        let text =
            "正常的前一句话。https://example.com/a/very/long/path/that/keeps/going。正常的当前句";
        let got = select_context(text, 64);
        assert!(got.contains("正常的当前句"));
        assert!(got.contains("正常的前一句话"));
        assert!(!got.contains("example.com"));
    }

    #[test]
    fn the_last_segment_is_kept_even_when_junky() {
        // 光标就在代码中间:当前段无论如何都要给模型(它是正在编辑的内容)
        let text = "上一句话。let x = some_function_call(argument)fn";
        let got = select_context(text, 64);
        assert!(got.contains("fn"));
        assert!(got.contains("上一句话"));
    }
}
