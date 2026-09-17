/// 一处编辑：把用户敲的串变成纠正后的串。下标是**纠正后**串里的位置（纯 ASCII 小写，字节即字符）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Edit {
    /// 原串这一位敲的是 `from`，纠正后换成了别的字母。
    Substitute { index: usize, from: char },

    /// 原串在这一位多敲了 `removed`，纠正后没有它（`index` 是它在纠正后串里本该在的位置）。
    Delete { index: usize, removed: char },

    /// 原串漏了一个字母，纠正后在 `index` 处多出一个。
    Insert { index: usize },

    /// 原串这一位与下一位敲反了（原来是 `first``second`，纠正后是 `second``first`）。
    Transpose {
        index: usize,
        first: char,
        second: char,
    },
}

impl Edit {
    /// 这处编辑落在（应用它之前的）串里的位置：换位取前一位，删除取被删的那位。
    pub fn position(&self) -> usize {
        match *self {
            Self::Substitute { index, .. }
            | Self::Insert { index }
            | Self::Transpose { index, .. }
            | Self::Delete { index, .. } => index,
        }
    }

    /// （本编辑应用前的）串里 `at` 处的间隙在（应用后的）串里落在哪：第二处编辑的画线位置
    /// 按它换算到最终串的坐标系。`at` 正好是被删字母的位置时取原地（画线语义是「画在这一位之前」）。
    pub fn map_gap_backward(&self, at: usize) -> usize {
        match *self {
            Self::Substitute { .. } | Self::Transpose { .. } => at,
            // 应用后串在 index 处少一个字母：之后的间隙往前挪一位
            Self::Delete { index, .. } => {
                if at > index {
                    at - 1
                } else {
                    at
                }
            }
            // 应用后串在 index 处多一个字母：之后的间隙往后挪一位
            Self::Insert { index } if at >= index => at + 1,
            Self::Insert { .. } => at,
        }
    }

    /// [`Self::map_gap_backward`] 的逆：应用后串里的间隙 `gap` 对应应用前串里的哪个间隙。
    /// 双错删除线换算到同一间隙时按原串里的先后排（两处编辑各自的被改字母在原串的位置）。
    pub fn gap_in_source(&self, gap: usize) -> usize {
        match *self {
            Self::Substitute { .. } | Self::Transpose { .. } => gap,
            // 应用后串在 index 处少一个字母：之后的间隙在原串里后移一位
            Self::Delete { index, .. } if gap > index => gap + 1,
            Self::Delete { .. } => gap,
            // 应用后串在 index 处多一个字母：之后的间隙在原串里前移一位
            Self::Insert { index } if gap > index => gap - 1,
            Self::Insert { .. } => gap,
        }
    }

    /// 纠正后串开头 `corrected_len` 个字母对应原串开头多少个字母：候选按纠正后的音节消耗拼音，
    /// 消耗掉的原串长度要按这处编辑换算回去。
    pub fn to_original(&self, corrected_len: usize) -> usize {
        match *self {
            Self::Substitute { .. } | Self::Transpose { .. } => corrected_len,
            // 多敲的字母紧贴在消耗掉的部分后面时一并吃掉，别把它留给下一段
            Self::Delete { index, .. } if index <= corrected_len => corrected_len + 1,
            Self::Delete { .. } => corrected_len,
            Self::Insert { index } if index < corrected_len => corrected_len - 1,
            Self::Insert { .. } => corrected_len,
        }
    }

    /// 要画删除线的原字母及其在纠正后串里的位置（画在这一位之前）；漏字没有可划的。
    pub fn struck(&self) -> Option<(usize, String)> {
        match *self {
            Self::Substitute { index, from } => Some((index, from.to_string())),
            Self::Delete { index, removed } => Some((index, removed.to_string())),
            Self::Insert { .. } => None,
            Self::Transpose {
                index,
                first,
                second,
            } => Some((index, format!("{first}{second}"))),
        }
    }
}

/// `input` 的全部一处编辑变体：先换位，再替换、删除、插入。`input` 必须是纯小写字母。
pub fn variants(input: &str) -> Vec<(Edit, String)> {
    let bytes = input.as_bytes();
    let n = bytes.len();
    let mut out = Vec::with_capacity(n * 55);
    for i in 0..n.saturating_sub(1) {
        if bytes[i] != bytes[i + 1] {
            let mut v = bytes.to_vec();
            v.swap(i, i + 1);
            out.push((
                Edit::Transpose {
                    index: i,
                    first: bytes[i] as char,
                    second: bytes[i + 1] as char,
                },
                String::from_utf8(v).expect("ascii"),
            ));
        }
    }
    for i in 0..n {
        for letter in b'a'..=b'z' {
            if letter == bytes[i] {
                continue;
            }
            let mut v = bytes.to_vec();
            v[i] = letter;
            out.push((
                Edit::Substitute {
                    index: i,
                    from: bytes[i] as char,
                },
                String::from_utf8(v).expect("ascii"),
            ));
        }
    }
    for i in 0..n {
        let mut v = bytes.to_vec();
        v.remove(i);
        out.push((
            Edit::Delete {
                index: i,
                removed: bytes[i] as char,
            },
            String::from_utf8(v).expect("ascii"),
        ));
    }
    for i in 0..=n {
        for letter in b'a'..=b'z' {
            let mut v = bytes.to_vec();
            v.insert(i, letter);
            out.push((
                Edit::Insert { index: i },
                String::from_utf8(v).expect("ascii"),
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_consumed_length_back_to_the_original() {
        assert_eq!(
            Edit::Substitute {
                index: 3,
                from: 'o'
            }
            .to_original(5),
            5
        );
        // 原 nihooma（7）→ 纠正 nihoma？这里只看换算：删掉下标 3 的字母，消耗 3 个纠正后字母时把它一起吃掉
        let delete = Edit::Delete {
            index: 3,
            removed: 'o',
        };
        assert_eq!(delete.to_original(2), 2);
        assert_eq!(delete.to_original(3), 4);
        assert_eq!(delete.to_original(6), 7);
        let insert = Edit::Insert { index: 3 };
        assert_eq!(insert.to_original(3), 3);
        assert_eq!(insert.to_original(4), 3);
        assert_eq!(insert.to_original(7), 6);
    }

    #[test]
    fn generates_every_single_edit_once() {
        let all = variants("ab");
        let texts: Vec<&str> = all.iter().map(|(_, t)| t.as_str()).collect();
        assert!(texts.contains(&"ba"));
        assert!(texts.contains(&"ac"));
        assert!(texts.contains(&"a"));
        assert!(texts.contains(&"zab"));
        // 1 换位 + 2×25 替换 + 2 删除 + 3×26 插入
        assert_eq!(all.len(), 1 + 50 + 2 + 78);
        assert!(
            variants("aa")
                .iter()
                .all(|(e, _)| !matches!(e, Edit::Transpose { .. }))
        );
    }
}
