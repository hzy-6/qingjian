# 生活闲聊词库：来源评估与合并记录（2026-09-18）

## 结论（最优解）

生活闲聊词的最优来源是**自建 + 用户整理**，不是大规模导入互联网词表：

- 现有 92,834 条基础词库已实质覆盖所有开放许可通用词表（jieba 34.9 万条放宽过滤后仅剩 75 条交集，且头部是 中国共产党/闯王/剩余价值 这类非闲聊词）；
- 开源词表对「到家了吗 / 吃了吗 / 回头见」这类整句组合覆盖几乎为零，该空间只能自建；
- 搜狗 .scel（版权）、Rime（GPL）、LCCC/STC（仅限研究）明确排除。

## 来源评估（随包分发视角）

| 来源 | 许可 | 结论 |
| --- | --- | --- |
| [jieba dict.txt](https://github.com/fxsjy/jieba) | MIT | 可用；34.9 万条带词频，作未来批量补充的词频基准（已下载验证，交集极小） |
| [phrase-pinyin-data](https://github.com/mozillazg/phrase-pinyin-data) | MIT | 可用；4.7 万短语分音节拼音，作拼音标注基准 |
| [pinyin-data](https://github.com/mozillazg/pinyin-data) | MIT | 可用；单字读音兜底 |
| [open-gram](https://github.com/sunpinyin/open-gram) | Apache-2.0 | 可用；词频交叉校准 |
| [libpinyin-dict](https://github.com/broly8/libpinyin-dict) | MPL-2.0 | 可用（文件级署名）；含「日常用语」分类 |
| [CC-CEDICT](https://www.mdbg.net/chinese/dictionary?page=cedict) | CC-BY-SA-4.0 | 需署名+该部分同方式共享，优先级低 |
| 搜狗 .scel | 版权不明 | 排除（参考 2007 谷歌输入法事件） |
| Rime luna-pinyin / 八股文 | GPL / LGPL | 排除（项目已因 GPL 移除雾凇数据） |
| LCCC / STC 对话语料 | 仅限研究 | 排除 |
| [hanzi-words-annual](https://github.com/zispace/hanzi-words-annual) | 无 LICENSE | 排除 |

## 合并记录

- 用户整理 642 条（`词库/生活闲聊词库_青简.tsv`）→ 173 条已在词库跳过、462 条经贪心最长切分 + 单字读音校验并入（9 处源拼音标注人工修正、ü 规范为 v）；
- 第二批自建 49 条常用语（好久不见 / 没问题 / 注意安全 / 新年快乐 等高频整块组合，自建规避许可问题），经同一校验管线并入；
- 第三批补录 6 条（别难过 / 有点难过 / 我发你了 / 必须安排 / 可能会晚点 / 慢慢过来，对抗审查发现的静默丢弃词）；
- 对抗审查发现：7 条被静默丢弃（贪心切分失败进复核清单后蒸发）→ 已补录 6 条（别难过/有点难过/我发你了/必须安排/可能会晚点/慢慢过来），QQ上说（拉丁字母词）明确不收；
- 173 条已存在常用语按 max(现频, 源频) 调频 47 行（请你吃饭 31→3300、微信 716→20000、乘凉 226→2000 等）；
- 嗯/诶 单字属词表例外（不在通用规范 8105 字表，但删掉会让 en/ei 无候选），保留并在此记录；
- 补充批量并入工具 `tools/lexicon-merge-check.py`（贪心切分 + 逐字读音校验 + 去重 + 复核处置清单）。
