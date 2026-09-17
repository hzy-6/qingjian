use crate::engine::{MarkedKind, MarkedSegment};
use crate::parser::Segmentation;

use super::edit::Edit;

/// 这处编辑用于找归属音节的位置：删除把多敲的字母算到它前面的音节上（与 [`Correction::typo_pair`] 一致），
/// 其余取编辑自身位置。
fn preceding_position(edit: &Edit) -> usize {
    match edit {
        Edit::Delete { index, .. } => index.saturating_sub(1),
        _ => edit.position(),
    }
}

/// 一次拼写纠正：用户敲的串、纠正后的串、那一处（或两处）编辑，以及纠正后串的完整音节切分。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Correction {
    /// 用户敲的（作用域里的拼音，不含 `'`）。
    pub original: String,

    /// 纠正后的拼音。
    pub corrected: String,

    /// 从原串到纠正后串的第一处编辑。
    pub edit: Edit,

    /// 第二处编辑（双错联合纠错才有）：坐标系是**第一处编辑应用后**的串，
    /// 它再改一处得到 `corrected`。单错时为 `None`。
    pub second: Option<Edit>,

    /// 纠正后串的切分，每个音节都完整。
    pub segmentation: Segmentation,
}

impl Correction {
    /// 纠正后串开头 `corrected_len` 个字母对应原串开头多少个字母：候选按纠正后的音节消耗拼音，
    /// 消耗掉的原串长度要按（两处）编辑依次换算回去。
    pub fn to_original(&self, corrected_len: usize) -> usize {
        if corrected_len == 0 {
            // 没消耗字母就什么都不对应；`Delete` 的「贴着消耗段的字母一并吃掉」在零点不成立
            return 0;
        }
        let after_first = match &self.second {
            Some(second) => second.to_original(corrected_len),
            None => corrected_len,
        };
        self.edit.to_original(after_first)
    }

    /// 这处编辑落在纠正后哪个音节里：返回 (敲的那段字母, 纠正后的音节)，给个人敲错表记（敲的那段多半不是合法音节）；
    /// 多敲的字母算在它前面那个音节上（与 [`Edit::to_original`] 一致）。编辑处不在任何音节里时返回 `None`。
    /// `consumed` 是上屏消耗掉的纠正后字母数：没吃到编辑处的上屏不算接受了纠正。
    pub fn typo_pair(&self, consumed: usize) -> Option<(String, String)> {
        let at = match self.edit {
            Edit::Substitute { index, .. }
            | Edit::Insert { index }
            | Edit::Transpose { index, .. } => index,
            Edit::Delete { index, .. } => index.saturating_sub(1),
        };
        if at >= consumed {
            return None;
        }
        let mut start = 0;
        for syllable in &self.segmentation.syllables {
            let end = start + syllable.text.len();
            if at < end {
                let typed_start = self.edit.to_original(start);
                let typed_end = self.edit.to_original(end);
                let typed = self.original.get(typed_start..typed_end)?;
                return (!typed.is_empty() && typed != syllable.text)
                    .then(|| (typed.to_owned(), syllable.text.clone()));
            }
            start = end;
        }
        None
    }

    /// 接受这次纠正记下的全部 (敲的, 要的) 音节对：第一处编辑一对，双错纠正的第二处编辑再来一对。
    /// 两处编辑都在最终串的坐标系里找各自落进的音节，敲的那段字母按两处编辑依次换算回原串取
    /// （第一处的位置先经 [`Edit::map_gap_backward`] 换到最终串）。两处编辑落进同一音节、
    /// 敲的段相同时只记一对。单错时就是 [`Self::typo_pair`]。
    pub fn typo_pairs(&self, consumed: usize) -> Vec<(String, String)> {
        let Some(second) = &self.second else {
            return self.typo_pair(consumed).into_iter().collect();
        };
        // 两处编辑复合的间隙换算：最终串 → 原串
        let to_source = |gap: usize| self.edit.to_original(second.to_original(gap));
        let mut pairs: Vec<(String, String)> = Vec::with_capacity(2);
        for at in [
            second.map_gap_backward(preceding_position(&self.edit)),
            preceding_position(second),
        ] {
            if let Some(pair) = self.pair_at(at, consumed, to_source)
                && !pairs.contains(&pair)
            {
                pairs.push(pair);
            }
        }
        pairs
    }

    /// `at`（最终串坐标）落进的那个音节的 (敲的那段字母, 音节)；音节区间经 `to_source` 换算回原串。
    fn pair_at(
        &self,
        at: usize,
        consumed: usize,
        to_source: impl Fn(usize) -> usize,
    ) -> Option<(String, String)> {
        if at >= consumed {
            return None;
        }
        let mut start = 0;
        for syllable in &self.segmentation.syllables {
            let end = start + syllable.text.len();
            if at < end {
                let typed_start = to_source(start);
                let typed_end = to_source(end);
                let typed = self.original.get(typed_start..typed_end)?;
                return (!typed.is_empty() && typed != syllable.text)
                    .then(|| (typed.to_owned(), syllable.text.clone()));
            }
            start = end;
        }
        None
    }

    /// preedit 的分段：纠正后的切分拼音（`'` 连接），被改掉的原字母以 [`MarkedKind::Corrected`] 插在它原来的位置。
    /// `nihooma` → `ni'h` + ~~o~~ + `ao'ma`；双错纠正按在最终串里的先后画两条删除线，
    /// 换算到同一间隙时再按原串里的先后排（被改的字母在原串哪个更靠前，哪条线先画）。
    pub fn marked_segments(&self) -> Vec<MarkedSegment> {
        let display = self.segmentation.joined("'");
        // (最终串里画线位置之前的字母数, 原串里被改字母的位置, 画线的字母)。插入编辑没有可画的字母。
        let mut strikes: Vec<(usize, usize, String)> = Vec::with_capacity(2);
        if let Some((_, struck)) = self.edit.struck() {
            // 第一处编辑的画线位置在「第二处编辑应用前」的坐标系里，换算到最终串与原串
            let at = match &self.second {
                Some(second) => second.map_gap_backward(self.edit.position()),
                None => self.edit.position(),
            };
            strikes.push((at, self.edit.gap_in_source(self.edit.position()), struck));
        }
        if let Some(second) = &self.second
            && let Some((at, struck)) = second.struck()
        {
            strikes.push((
                at,
                self.edit.gap_in_source(second.gap_in_source(at)),
                struck,
            ));
        }
        if strikes.is_empty() {
            return vec![MarkedSegment::new(display, MarkedKind::Typed)];
        }
        strikes.sort_by_key(|(at, in_original, _)| (*at, *in_original));
        let mut segments = Vec::new();
        let mut typed_start = 0;
        let mut byte = 0;
        let mut letters = 0;
        for (at, _, struck) in strikes {
            // 推进到第 at 个字母之前的间隙，途中的 `'` 留在前一段里
            while byte < display.len() {
                if display.as_bytes()[byte] == b'\'' {
                    byte += 1;
                    continue;
                }
                if letters == at {
                    break;
                }
                letters += 1;
                byte += 1;
            }
            if byte > typed_start {
                segments.push(MarkedSegment::new(
                    &display[typed_start..byte],
                    MarkedKind::Typed,
                ));
            }
            segments.push(MarkedSegment::new(struck, MarkedKind::Corrected));
            typed_start = byte;
        }
        if typed_start < display.len() {
            segments.push(MarkedSegment::new(
                &display[typed_start..],
                MarkedKind::Typed,
            ));
        }
        segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Syllable;

    fn segmentation(syllables: &[&str]) -> Segmentation {
        Segmentation {
            syllables: syllables.iter().map(|s| Syllable::complete(s)).collect(),
        }
    }

    fn texts(segments: &[MarkedSegment]) -> Vec<(String, MarkedKind)> {
        segments.iter().map(|s| (s.text.clone(), s.kind)).collect()
    }

    #[test]
    fn strikes_the_replaced_letter_in_place() {
        let correction = Correction {
            original: "nihooma".into(),
            corrected: "nihaoma".into(),
            edit: Edit::Substitute {
                index: 3,
                from: 'o',
            },
            second: None,
            segmentation: segmentation(&["ni", "hao", "ma"]),
        };
        assert_eq!(
            texts(&correction.marked_segments()),
            [
                ("ni'h".to_owned(), MarkedKind::Typed),
                ("o".to_owned(), MarkedKind::Corrected),
                ("ao'ma".to_owned(), MarkedKind::Typed),
            ]
        );
    }

    #[test]
    fn typo_pair_is_the_syllable_around_the_edit() {
        let correction = Correction {
            original: "nihooma".into(),
            corrected: "nihaoma".into(),
            edit: Edit::Substitute {
                index: 3,
                from: 'o',
            },
            second: None,
            segmentation: segmentation(&["ni", "hao", "ma"]),
        };
        assert_eq!(
            correction.typo_pair(7),
            Some(("hoo".to_owned(), "hao".to_owned()))
        );
        assert_eq!(
            correction.typo_pairs(7),
            vec![("hoo".to_owned(), "hao".to_owned())]
        );
        // 只吃到 ni 就上屏：没接受纠正
        assert_eq!(correction.typo_pair(2), None);
        let correction = Correction {
            original: "meiganxi".into(),
            corrected: "meiguanxi".into(),
            edit: Edit::Insert { index: 4 },
            second: None,
            segmentation: segmentation(&["mei", "guan", "xi"]),
        };
        assert_eq!(
            correction.typo_pair(9),
            Some(("gan".to_owned(), "guan".to_owned()))
        );
        let correction = Correction {
            original: "zhegge".into(),
            corrected: "zhege".into(),
            edit: Edit::Delete {
                index: 3,
                removed: 'g',
            },
            second: None,
            segmentation: segmentation(&["zhe", "ge"]),
        };
        // 多敲的 g 算在前一个音节上
        assert_eq!(
            correction.typo_pair(5),
            Some(("zheg".to_owned(), "zhe".to_owned()))
        );
    }

    #[test]
    fn extra_letter_at_the_end_and_missing_letter() {
        let correction = Correction {
            original: "nihaox".into(),
            corrected: "nihao".into(),
            edit: Edit::Delete {
                index: 5,
                removed: 'x',
            },
            second: None,
            segmentation: segmentation(&["ni", "hao"]),
        };
        assert_eq!(
            texts(&correction.marked_segments()),
            [
                ("ni'hao".to_owned(), MarkedKind::Typed),
                ("x".to_owned(), MarkedKind::Corrected),
            ]
        );
        let correction = Correction {
            original: "nhao".into(),
            corrected: "nihao".into(),
            edit: Edit::Insert { index: 1 },
            second: None,
            segmentation: segmentation(&["ni", "hao"]),
        };
        assert_eq!(
            texts(&correction.marked_segments()),
            [("ni'hao".to_owned(), MarkedKind::Typed)]
        );
    }

    /// 双错纠正：两处敲错各画一条删除线、各记一对 (敲的, 要的)，消耗长度按两处编辑依次换算。
    #[test]
    fn double_correction_carries_two_edits() {
        // 敲 jintainheh（jin tain heh），第一处换位 tain→tian，第二处把末尾的 h 换成 n：heh→hen
        let correction = Correction {
            original: "jintainheh".into(),
            corrected: "jintianhen".into(),
            edit: Edit::Transpose {
                index: 6,
                first: 'a',
                second: 'n',
            },
            second: Some(Edit::Substitute {
                index: 9,
                from: 'h',
            }),
            segmentation: segmentation(&["jin", "tian", "hen"]),
        };
        assert_eq!(
            correction.typo_pairs(10),
            [
                ("tain".to_owned(), "tian".to_owned()),
                ("heh".to_owned(), "hen".to_owned()),
            ]
        );
        // 画线按最终串里的先后：换位画 an（tian 末位之前），换键画 h（hen 末位之前）
        assert_eq!(
            texts(&correction.marked_segments()),
            [
                ("jin'tia".to_owned(), MarkedKind::Typed),
                ("an".to_owned(), MarkedKind::Corrected),
                ("n'he".to_owned(), MarkedKind::Typed),
                ("h".to_owned(), MarkedKind::Corrected),
                ("n".to_owned(), MarkedKind::Typed),
            ]
        );
        // 换位不改长度、替换也不改，消耗多少原串就是多少
        assert_eq!(correction.to_original(10), 10);
        assert_eq!(correction.to_original(6), 6);
        assert_eq!(correction.to_original(0), 0);
    }

    /// 双错里第一处是插入 / 第二处是删除时，第二处的坐标与画线位置要跨编辑换算。
    #[test]
    fn double_edits_across_length_changes() {
        // 敲 nihoxma（ni ho x ma），第一处补回 hao 漏的 a，第二处删掉多敲的 x → nihaoma
        let correction = Correction {
            original: "nihoxma".into(),
            corrected: "nihaoma".into(),
            edit: Edit::Insert { index: 3 },
            second: Some(Edit::Delete {
                index: 5,
                removed: 'x',
            }),
            segmentation: segmentation(&["ni", "hao", "ma"]),
        };
        assert_eq!(correction.to_original(7), 7);
        // 两处编辑都落在 hao 上，敲的段都是 hox：按两处编辑复合换算后只记一对
        assert_eq!(
            correction.typo_pairs(7),
            vec![("hox".to_owned(), "hao".to_owned())]
        );
        // 插入没有可画的字母；删掉的 x 画在 hao 与 ma 的间隙（含前面的 '）
        assert_eq!(
            texts(&correction.marked_segments()),
            [
                ("ni'hao'".to_owned(), MarkedKind::Typed),
                ("x".to_owned(), MarkedKind::Corrected),
                ("ma".to_owned(), MarkedKind::Typed),
            ]
        );
    }

    /// 第二处编辑在第一处**之前**删字母时，第一处的位置要经第二处换算到最终串：
    /// 归属音节、敲的段都不能错位（对抗审查找到的反例，曾记成 ("sha","hao")）。
    #[test]
    fn first_pair_crosses_the_second_delete() {
        // 敲 nxshaoma：s→i（位置 2，中间串坐标）+ 删掉多敲的 x（位置 1）→ nihaoma
        let correction = Correction {
            original: "nxshaoma".into(),
            corrected: "nihaoma".into(),
            edit: Edit::Substitute {
                index: 2,
                from: 's',
            },
            second: Some(Edit::Delete {
                index: 1,
                removed: 'x',
            }),
            segmentation: segmentation(&["ni", "hao", "ma"]),
        };
        // 两处编辑都落在 ni 上，敲的段都是 nxs：只记一对
        assert_eq!(
            correction.typo_pairs(7),
            vec![("nxs".to_owned(), "ni".to_owned())]
        );
        // 第一处在中间串末尾（位置 7）也不丢：换算到最终串 6，落在 ma 上
        let correction = Correction {
            original: "nxihaomk".into(),
            corrected: "nihaoma".into(),
            edit: Edit::Substitute {
                index: 7,
                from: 'k',
            },
            second: Some(Edit::Delete {
                index: 1,
                removed: 'x',
            }),
            segmentation: segmentation(&["ni", "hao", "ma"]),
        };
        assert_eq!(
            correction.typo_pairs(7),
            vec![
                ("mk".to_owned(), "ma".to_owned()),
                ("nxi".to_owned(), "ni".to_owned()),
            ]
        );
        // 两条删除线换算到同一间隙时按原串先后排：x（原串 1）在 k（原串 2）之前
        let correction = Correction {
            original: "nxkhaoma".into(),
            corrected: "nihaoma".into(),
            edit: Edit::Substitute {
                index: 2,
                from: 'k',
            },
            second: Some(Edit::Delete {
                index: 1,
                removed: 'x',
            }),
            segmentation: segmentation(&["ni", "hao", "ma"]),
        };
        assert_eq!(
            texts(&correction.marked_segments()),
            [
                ("n".to_owned(), MarkedKind::Typed),
                ("x".to_owned(), MarkedKind::Corrected),
                ("k".to_owned(), MarkedKind::Corrected),
                ("i'hao'ma".to_owned(), MarkedKind::Typed),
            ]
        );
    }

    /// 两处都是漏字母（第二处编辑当前生成器不出，坐标换算仍须正确）：各按各的音节记对，
    /// 没有可画的删除线。
    #[test]
    fn double_insert_edits_map_their_own_syllables() {
        // 敲 nha：补 i 成 niha，再补 o 成 nihao
        let correction = Correction {
            original: "nha".into(),
            corrected: "nihao".into(),
            edit: Edit::Insert { index: 1 },
            second: Some(Edit::Insert { index: 4 }),
            segmentation: segmentation(&["ni", "hao"]),
        };
        assert_eq!(correction.to_original(5), 3);
        assert_eq!(
            correction.typo_pairs(5),
            vec![
                ("n".to_owned(), "ni".to_owned()),
                ("ha".to_owned(), "hao".to_owned()),
            ]
        );
        assert_eq!(
            texts(&correction.marked_segments()),
            [("ni'hao".to_owned(), MarkedKind::Typed)]
        );
    }
}
