//! 神经重排池的文本多样性短名单。

use std::collections::HashSet;

use crate::sentence::Conversion;

/// 静态最稳的前几条无条件保留，剩余名额优先给相对首选具有新分歧位置组合的路径。
/// 例如「三教…技术」的前缀分歧「三角…技术」与后缀分歧「三教…及数」都在时，
/// 让「三角…及数」这类组合候选也有机会进入固定 8 条的 2B 池。
pub(super) fn shortlist(paths: Vec<Conversion>, limit: usize) -> Vec<Conversion> {
    if paths.len() <= limit || limit == 0 {
        return paths.into_iter().take(limit).collect();
    }
    let reference = paths[0].text.clone();
    let stable = limit.min(5);
    let mut selected: Vec<bool> = vec![false; paths.len()];
    let mut signatures = HashSet::new();
    for (index, path) in paths.iter().take(stable).enumerate() {
        selected[index] = true;
        signatures.insert(difference_signature(&reference, &path.text));
    }
    let mut count = stable;
    for (index, path) in paths.iter().enumerate().skip(stable) {
        if count >= limit {
            break;
        }
        if signatures.insert(difference_signature(&reference, &path.text)) {
            selected[index] = true;
            count += 1;
        }
    }
    // 分歧类型不足时按原始静态顺序补齐。
    for slot in &mut selected {
        if count >= limit {
            break;
        }
        if !*slot {
            *slot = true;
            count += 1;
        }
    }
    paths
        .into_iter()
        .zip(selected)
        .filter_map(|(path, keep)| keep.then_some(path))
        .collect()
}

fn difference_signature(reference: &str, candidate: &str) -> Vec<usize> {
    let reference: Vec<char> = reference.chars().collect();
    let candidate: Vec<char> = candidate.chars().collect();
    let length = reference.len().max(candidate.len());
    (0..length)
        .filter(|&index| reference.get(index) != candidate.get(index))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sentence::SentenceWord;

    fn path(text: &str, score: f64) -> Conversion {
        Conversion {
            text: text.to_owned(),
            syllables: Vec::new(),
            words: vec![SentenceWord {
                text: text.to_owned(),
                syllables: Vec::new(),
                placeholder: false,
            }],
            score,
            static_score: score,
            penalty: 0.0,
        }
    }

    #[test]
    fn keeps_five_stable_paths_then_prefers_new_difference_combinations() {
        let paths = vec![
            path("甲乙丙丁", 16.0),
            path("戊乙丙丁", 15.0),
            path("甲乙丙己", 14.0),
            path("甲庚丙丁", 13.0),
            path("甲乙辛丁", 12.0),
            path("壬乙丙丁", 11.0), // 与第 2 条同为仅首字分歧
            path("甲癸丙丁", 10.0), // 与第 4 条同为仅第二字分歧
            path("戊乙丙己", 9.0),  // 前后两处组合分歧
            path("甲庚辛丁", 8.0),  // 中间两处组合分歧
        ];
        let texts: Vec<String> = shortlist(paths, 7)
            .into_iter()
            .map(|path| path.text)
            .collect();
        assert_eq!(
            texts,
            [
                "甲乙丙丁",
                "戊乙丙丁",
                "甲乙丙己",
                "甲庚丙丁",
                "甲乙辛丁",
                "戊乙丙己",
                "甲庚辛丁"
            ]
        );
    }
}
