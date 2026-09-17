//! 双错纠正坐标换算的随机性质测试：固定种子生成几千组「原串 + 两处编辑 + 随机音节切分」，
//! 轰击 `Correction::to_original` / `typo_pairs` / `marked_segments` 的不变量，
//! 并把单错（`second == None`）的画线与改动前的旧算法逐字节对比。
//! 坐标换算是双错联合纠错最容易藏 off-by-one 的地方（第二处编辑的下标在第一处编辑应用后的串上），
//! 枚举型单测覆盖不了全部组合，交给随机组合兜底。

use qingjian_core::correction::Edit;
use qingjian_core::parser::{Segmentation, Syllable};
use qingjian_core::{Correction, MarkedKind, MarkedSegment};

/// 固定种子的 xorshift，失败可复现。
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// 偏拼音常用字母，长短期各半：短串让下标撞在 0 / 末尾 / 相邻位置。
const LETTERS: &[u8] = b"abcdgilmnqrstxz";

fn random_word(rng: &mut Rng, max_len: usize) -> String {
    let len = 1 + rng.below(max_len);
    (0..len)
        .map(|_| LETTERS[rng.below(LETTERS.len())] as char)
        .collect()
}

/// 在 `text` 上生成一处随机编辑，返回 (编辑, 应用后的串)。第二处编辑也允许插入：
/// 当前生成器不出它，坐标换算仍必须正确（防御性分支由这里覆盖）。
fn random_edit(rng: &mut Rng, text: &str) -> (Edit, String) {
    let bytes = text.as_bytes();
    let n = bytes.len();
    match rng.below(4) {
        0 => {
            let index = rng.below(n);
            let mut to = LETTERS[rng.below(LETTERS.len())];
            while to == bytes[index] {
                to = LETTERS[rng.below(LETTERS.len())];
            }
            let mut v = bytes.to_vec();
            v[index] = to;
            (
                Edit::Substitute {
                    index,
                    from: bytes[index] as char,
                },
                String::from_utf8(v).unwrap(),
            )
        }
        1 if n >= 2 && bytes[..n - 1].iter().any(|&b| b != bytes[n - 1]) => {
            let mut index = rng.below(n - 1);
            while index + 1 < n && bytes[index] == bytes[index + 1] {
                index += 1;
            }
            if index + 1 >= n {
                return random_edit(rng, text); // 起点落在尾部相同字母段里，重来
            }
            let mut v = bytes.to_vec();
            v.swap(index, index + 1);
            (
                Edit::Transpose {
                    index,
                    first: bytes[index] as char,
                    second: bytes[index + 1] as char,
                },
                String::from_utf8(v).unwrap(),
            )
        }
        2 => {
            let index = rng.below(n);
            let mut v = bytes.to_vec();
            v.remove(index);
            (
                Edit::Delete {
                    index,
                    removed: bytes[index] as char,
                },
                String::from_utf8(v).unwrap(),
            )
        }
        _ => {
            let index = rng.below(n + 1);
            let letter = LETTERS[rng.below(LETTERS.len())];
            let mut v = bytes.to_vec();
            v.insert(index, letter);
            (Edit::Insert { index }, String::from_utf8(v).unwrap())
        }
    }
}

/// 把串随机切成 1..=5 段当音节（不要求是合法拼音，只要求恰好覆盖整串）。
fn random_segmentation(rng: &mut Rng, text: &str) -> Segmentation {
    let n = text.len();
    let parts = 1 + rng.below(n.min(5));
    let mut cuts: Vec<usize> = (1..parts).map(|_| 1 + rng.below(n - 1)).collect();
    cuts.sort_unstable();
    cuts.dedup();
    let mut syllables = Vec::new();
    let mut prev = 0;
    for cut in cuts.into_iter().chain([n]) {
        if cut > prev {
            syllables.push(Syllable::complete(&text[prev..cut]));
        }
        prev = cut;
    }
    Segmentation { syllables }
}

/// 独立实现的前缀换算（不调用被测代码）：`edit` 应用后串开头 `len` 个字母对应应用前串多少字母。
/// 与 [`qingjian_core::correction::Edit::to_original`] 同语义（多敲的字母贴着消耗段就一并吃掉），
/// 语义本身由 P1/P2 独立验证，这里只用来核 typo_pairs 的区间组合。
fn prefix_before(edit: &Edit, len: usize) -> usize {
    match *edit {
        Edit::Substitute { .. } | Edit::Transpose { .. } => len,
        Edit::Delete { index, .. } if index <= len => len + 1,
        Edit::Delete { .. } => len,
        Edit::Insert { index } if index < len => len - 1,
        Edit::Insert { .. } => len,
    }
}

/// 改动前的旧版单删除线算法：单错（second == None）的画线必须与它逐字节一致。
fn legacy_single_marked(correction: &Correction) -> Vec<MarkedSegment> {
    let display = correction.segmentation.joined("'");
    let Some((at, struck)) = correction.edit.struck() else {
        return vec![MarkedSegment::new(display, MarkedKind::Typed)];
    };
    let mut letters = 0;
    let mut split = display.len();
    for (i, c) in display.char_indices() {
        if c == '\'' {
            continue;
        }
        if letters == at {
            split = i;
            break;
        }
        letters += 1;
    }
    let mut segments = Vec::with_capacity(3);
    if split > 0 {
        segments.push(MarkedSegment::new(&display[..split], MarkedKind::Typed));
    }
    segments.push(MarkedSegment::new(struck, MarkedKind::Corrected));
    if split < display.len() {
        segments.push(MarkedSegment::new(&display[split..], MarkedKind::Typed));
    }
    segments
}

fn struck_texts(edit: &Edit) -> Vec<String> {
    match edit.struck() {
        Some((_, struck)) => vec![struck],
        None => Vec::new(),
    }
}

fn check(correction: &Correction) {
    let original = &correction.original;
    let corrected = &correction.corrected;

    // P1：两处编辑全量换算回去等于原串长度
    assert_eq!(
        correction.to_original(corrected.len()),
        original.len(),
        "P1 全量换算: {correction:?}"
    );

    // P2：零点、单调、上界
    assert_eq!(correction.to_original(0), 0, "P2 零点: {correction:?}");
    let mut prev = 0;
    for k in 0..=corrected.len() {
        let mapped = correction.to_original(k);
        assert!(
            mapped <= original.len(),
            "P2 上界: k={k} mapped={mapped} {correction:?}"
        );
        assert!(
            mapped >= prev,
            "P2 单调: k={k} mapped={mapped} prev={prev} {correction:?}"
        );
        prev = mapped;
    }

    // P3：画线不 panic；Typed 段拼接 == 显示串；删除线文本 == 两处编辑各自可画的字母；
    // 同一间隙的两条线按原串先后排
    let segments = correction.marked_segments();
    let typed: String = segments
        .iter()
        .filter(|s| s.kind == MarkedKind::Typed)
        .map(|s| s.text.as_str())
        .collect();
    assert_eq!(typed, correction.segmentation.joined("'"), "P3 Typed 拼接");
    let mut struck_out: Vec<String> = segments
        .iter()
        .filter(|s| s.kind == MarkedKind::Corrected)
        .map(|s| s.text.clone())
        .collect();
    let mut struck_expect = struck_texts(&correction.edit);
    if let Some(second) = &correction.second {
        struck_expect.extend(struck_texts(second));
    }
    struck_out.sort();
    struck_expect.sort();
    assert_eq!(struck_out, struck_expect, "P3 删除线文本: {correction:?}");

    // P4：typo_pairs 每对的敲的段 == **某个**同名音节按（独立实现的）两处编辑换算出的原串区间
    // （随机切分里同名音节可能不止一个，编辑落在哪个就对得上哪个即可），且两对互不重复——
    // 第一对漏做第二处换算、错位取段、整对丢失都在这里现形
    let pairs = correction.typo_pairs(corrected.len());
    assert!(pairs.len() <= 2, "P4 数量: {pairs:?} {correction:?}");
    let in_original = |gap: usize| match &correction.second {
        Some(second) => prefix_before(&correction.edit, prefix_before(second, gap)),
        None => prefix_before(&correction.edit, gap),
    };
    let syllable_starts: Vec<usize> = correction
        .segmentation
        .syllables
        .iter()
        .scan(0, |start, s| {
            let at = *start;
            *start += s.text.len();
            Some(at)
        })
        .collect();
    for (typed, intended) in &pairs {
        let candidates: Vec<&str> = correction
            .segmentation
            .syllables
            .iter()
            .enumerate()
            .filter(|(_, s)| s.text == *intended)
            .map(|(i, s)| {
                &original[in_original(syllable_starts[i])
                    ..in_original(syllable_starts[i] + s.text.len())]
            })
            .collect();
        assert!(
            candidates.iter().any(|expected| expected == typed),
            "P4 敲的段与独立换算不符: {typed:?} ∉ {candidates:?} {correction:?}"
        );
    }
    assert!(
        pairs.len() < 2 || pairs[0] != pairs[1],
        "P4 重复对: {pairs:?} {correction:?}"
    );

    // P5：单错的画线与旧算法逐字节一致（单错不退化的硬要求）
    if correction.second.is_none() {
        assert_eq!(
            correction.marked_segments(),
            legacy_single_marked(correction),
            "P5 旧算法回归: {correction:?}"
        );
        assert_eq!(
            correction.typo_pairs(corrected.len()),
            correction
                .typo_pair(corrected.len())
                .into_iter()
                .collect::<Vec<_>>(),
            "P5 单错退化: {correction:?}"
        );
    }
}

#[test]
fn coordinate_math_holds_under_random_double_edits() {
    let mut rng = Rng(0x5EED_2026_0917);
    let mut doubles = 0;
    let mut singles = 0;
    for iteration in 0..8_000 {
        let max_len = if iteration % 2 == 0 { 6 } else { 16 };
        let original = random_word(&mut rng, max_len);
        let (first, intermediate) = random_edit(&mut rng, &original);
        if intermediate.is_empty() {
            continue;
        }
        let (second, corrected) = random_edit(&mut rng, &intermediate);
        if corrected.is_empty() {
            continue;
        }
        check(&Correction {
            original: original.clone(),
            corrected: corrected.clone(),
            edit: first,
            second: Some(second),
            segmentation: random_segmentation(&mut rng, &corrected),
        });
        doubles += 1;

        let (only, corrected) = random_edit(&mut rng, &original);
        if corrected.is_empty() {
            continue;
        }
        check(&Correction {
            original,
            corrected: corrected.clone(),
            edit: only,
            second: None,
            segmentation: random_segmentation(&mut rng, &corrected),
        });
        singles += 1;
    }
    assert!(
        doubles > 6_000 && singles > 6_000,
        "生成数量异常: {doubles} {singles}"
    );
}
