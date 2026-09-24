# 各 crate 的实现要点

CLAUDE.md 只保留目录地图与规则，每个 crate / app / tool 的实现细节收在这里：入口类型、数据文件、常数、生成命令。
改了实现要同步改这里；与代码冲突时以代码为准。

## crates/qingjian-dictionary

词库（TSV 解析或 `.qj` mmap），键按字节序排好，查询逐音节位置二分收窄（简拼位置按音节块跳扫），
`lookup_pattern`（≥ 模式长度）与 `lookup_exact`（正好等长）同一套实现。词库键以 `v` 表示 ü，
TSV 解析、查询与生成工具把 `lue` / `nue` 统一成 `lve` / `nve`。
旧 `.qj` 含这些键时，加载器建立规范化的内存词库。新 `.qj` 继续使用 mmap。

## crates/qingjian-core

模块：`composition` / `parser` / `correction`（拼写纠错：整段一两处编辑的候选纠正——单错压不过原样时再叠一处高频编辑联合纠错，`second_variants` 只出换位 / 相邻键替换 / 多敲 + `typo` 音节级敲错变体表，后者进整句词图当带代价的边）/
`candidate` / `ranking` / `shortcut` / `sentence` / `fuzzy` / `shuangpin`（双拼：四套方案键位表、键 → 全拼解码与消耗换算）/ `zhuyin`（大千注音：键 → 注音符号 → 拼音，`[general] zhuyin` 开关，声调只判音节完整不进查询）/ `emoji` /
`english`（英文模式候选）/ `engine`（`query::EnglishTail`：句末英文词并入整句，`woxiangxuehaorust` → 我想学好rust，尾段也像拼音时按分数与拼音读法比）。
`Engine` 是对外唯一门面，`Translator` / `Learner` trait 在 `engine` 模块；词库是「主词库 + 附加词库（`set_extra_dictionaries`）+ 用户词」的列表；评测发现的常用词缺口按 `domain_words.tsv` 规范人工补过一批
（会议室 / 这辆 / 母亲河 / 吓倒 / 三份 / 各部门 / 带货 / 银杏叶 / 轻拂，另调 微信 / 乘凉 频次）；生活闲聊词库 462 条（桌面 词库/生活闲聊词库_青简.tsv，贪心最长切分 + 单字读音校验，修正 9 处源文件拼音标注）并入 domain_words.tsv 与 dict.tsv；剩余 22 条 miss 的根因结论与复开条件见 [eval-miss-closeout.md](eval-miss-closeout.md)；生活闲聊词库 462+49+6 条并入选源评估见 [chat-lexicon.md](chat-lexicon.md)；随包模型换为 uer/gpt2-chinese-cluecorpussmall 102M 转制版（Apache-2.0，加载器新增可选 `head.weight` 支持不共享输出头的模型）；繁体输出（`traditional` 开关与 `traditional_map` 映射）依赖 `ferrous-opencc`（`s2tw`）在出候选与上屏边界转换，内部保持简体。
`Engine` 是对外唯一门面，`Translator` / `Learner` trait 在 `engine` 模块；词库是「主词库 + 附加词库（`set_extra_dictionaries`）+ 用户词」的列表。
- `sentence::convert_paths` 在已有两个完整音节时将末尾单字母作为前缀参与整句转换（`nihaom` 可由「你好」+「吗」组句），单个完整音节后仍等末尾至少两个字母，避免首键高开销。
- 整句候选的跨切分仲裁：parser 首切是「音节少、前面音节长」的贪心结果（`bange` → `bang e`），语言模型时常更认可另一支（`ban ge` → 半个小时，整句评测上从 67.6% 提到 69.8%）。
  `plain_sentence` 对前 `SENTENCE_SEGMENTATIONS` = 4 个切分各做一次静态整句转换（`convert_sentence_raw`，不重排），赢家再单独做神经重排（各切分都重排太贵）。
  仲裁三道闸：两边音节全完整且音节数一样（含简拼 / 前缀的切分分数没法比）、对手要赢出 `SENTENCE_ARBITRATION_MARGIN` = 2.5 nat（个人 n-gram / 敲错折扣喂出来的小分差不作数）、
  首切的最优路径没走敲错 / 模糊边（`nineng` 按 `nin eng` + 个人敲错表是用户自己的读法，不让干净切分压掉）。
  末闸有一个例外：对手是整段输入的那一个词库词、且按原样读（`chuanganqi` → 传感器；首切的 `chuang an qi` 靠 an→kan 敲错边读出 创刊起）时，
  整词是「用户把整个词拼出来了」的硬信号，允许它按分数翻掉改了原样的首切，免得最后给首切套一个干净但荒唐的 床安琪 压住 传感器。回归测试在 `engine/tests/lookup.rs`。
- 字符级整句提议器（`sentence::CharacterProposer`，实现在 `qingjian-lm` 的 `CharNgramModel`，可 mmap 的字符 n-gram FST）：
  对胜出切分的每个音节取词库前 `CHARACTER_CHOICES`=50 个同音字，用字 n-gram 按拼音约束解出整句字串，与词级前 8 条路径合池，
  再由重排器预选回 `neural_paths`（缺省 8）条（旧静态首选钉住）。合池在**静态切分仲裁之后、神经重排之前**、只对胜出切分做一次（每切分都做会把 139 句 P50 从 186ms 拉到 747ms）。
  同步打分器当场打原始分；异步打分器分未到时先整池留着交给 `rescore_paths` want，分到齐后在 `preselect_character_paths` 按缓存原始分预选（同步 / 异步结果一致）。
  CLI 用 `--char-model data/generated/char5.fst` 开；macOS 壳在 `[model] enabled` 下从用户目录 / 包内 `model/char5.fst` 加载。
  实测（Qwen 2B + 成语 + IT）：139 集首选 93.5%→95.7%、oracle 95.0%→97.1%；895 混域集首选 61.9%→68.3%、oracle 68.5%→79.5%、字准确率 91.0%→92.9%。
- 中英混输的英文词位置：`Engine::set_chinese_first`（配置 `[general] chinese_first`，缺省开；CLI 不读配置，`--chinese-first` 才开）关着时拼音不像话的输入英文排第一
  （`extras::insert_english`，用户老选中文词时仍让中文在前），开着时整句先插、英文词紧随其后排第二（`query_inner` 里两步的先后按开关掉转）；句末英文词并入整句（`EnglishTail`）不受它影响。
  早期缺省关是回放定的（9241 词 / 269 条英文上屏：缺省开英文首选 82.5% → 7.1%），后来改成缺省开；`leis` / `hz` / `bus` / `key` 四个不像拼音的输入两种排法都有回归测试钉着（`engine/tests/english.rs`）。
- `custom_phrase::merge_replacements` 把平台给的「输入码 → 短语」表（macOS 系统文本替换）并进配置里的自定义短语：每条占该码最靠前的空位（1–9），
  输入码不是小写字母、已有同码同文本、九位都满的跳过；Core 不管数据从哪来。

`EngineSession` 保存可挂起的组句、标点、历史与学习链，`Engine::swap_session` 在同一个引擎里交换输入状态，共用词库与落盘服务。切换上下文时清除查询及异步预测缓存，并由平台恢复各自私密状态。

`Engine::discard_input` / `EngineSession::discard_input` 用于隐私能力变化时无痕清理输入，包括透传缓冲、学习链和暂存词汇曝光；`set_private` 只切换写入开关，保留已输入的组句。

## crates/qingjian-translate

`Glossary`，本地 TSV 释义表（词性 + 译文）；`LevelTable`，词汇等级表（`assets/levels/levels-{en,ja}.tsv`，CEFR A1–C2 / JLPT N5–N1，
`uv run tools/corpus/levels.py` 从 `data/levels/` 的原始 CSV 生成，来源与许可见 `assets/levels/README.md`），「统计」页按级数词汇用，不进候选。

## crates/qingjian-learning

- `FrequencyLearner`：用户选择次数（`user.tsv`）、按输入串记的选择（`user-choices.tsv`，词级排序里同输入串选过的优先）、用户词（`user-words.tsv`，主词库同格式，
  Engine 与主词库一起查）、个人英文词（`user-english.tsv`，回车原样上屏的英文词与选过的英文候选，与随包英文词表一起出候选且在前）、
  个人敲错表（`user-typos.tsv`，接受过的 (敲的, 要的) 音节对，词图敲错边与整段纠错的代价按它打折）与个人 n-gram（`user-ngram.tsv`，Core `sentence::UserNgram`，
  二元 + 三元在线计数，整句转换与词级排序里与静态模型插值；Tab 接受的云端整句按 `sentence::segment_text` 切词后也记；
  连着选出的两个词记够次数自动造词进用户词，一段拼音分几次选完的合成词记两次也造）。
- `InputLog`：输入日志（`input-log.jsonl`，每次上屏一行：敲的键、切分、看到的前几个候选、选了第几个、来源、纠错、撤销，
  Core `InputLogger` trait 的落盘实现，`[general] input_log` 缺省开，只写本机，给离线回归评测与个人模型用）。
- `UsageStats`：输入统计（`usage.tsv`，按天记汉字 / 中文词 / 英文词 / 上屏次数，Core `UsageMeter` trait 的实现，Engine 每次上屏 `Usage::of_text` + 按来源定词数，
  整句按 `segment_text` 切词数；与输入日志无关，偏好设置「统计」页显示，`book_scale` 折成几本《某书》）。
- `VocabularyBook`：词汇记录（`user-vocab.tsv`，Core `VocabularyTracker` trait 的实现：学习语言的每条译词看到过几轮 / 上屏过 / ⌥+数字 打出过几次；Core 私密输入统一跳过曝光和提交写入，但仍可读取已有记录用于排序和生词标记；
  Engine `annotate` 据此填 `Sense::fresh`，看到轮次不到 `FRESH_UNTIL` = 3 的译词壳里画橙色；「看到」按上屏那一刻屏幕上那一页算，壳每次画完 `Engine::note_displayed` 告知当前页）。
- 各表落盘走 Core `storage::write_atomic`（临时文件 + fsync + 改名），加载按行容错（坏行警告跳过，真读不了壳退回内存学习），
  壳激活期间每 60 秒 `Engine::flush_learning`；IMK 回调边界 `imk::catch_panic` 拦 panic、缓冲区字母原样上屏（见 architecture.md「崩溃不丢」）。

## crates/qingjian-predict

- `CloudPredictor`：`Predictor` trait 的网络实现（async-openai，OpenAI 兼容接口，默认 DeepSeek），后台线程防抖 / 缓存 / 超时，`submit` / `poll` 非阻塞。
  `PredictConfig` 是配置的 `[predict]` 分节（`local` 是本地整句联想开关，缺省开，实现在 Core）。只在组句中联想，一次请求给云端词（容错校验后补进候选第一页末尾 `[predict] slots` 格，缺省 2，不预留不占位，
  前面的本地候选不挪；排布在 Core `CandidateLayout`）和整句补全（preedit 右侧，Tab）；上屏后不联想，本地历史不进请求。
- `CloudGlossFiller`：释义兜底（Core `GlossFiller` trait，与 Predictor 分开的线程与通道，攒 1.5 秒 / 8 个词发一次，问过不再问）：
  随包释义表没有的词库词 / 云端词上屏后入队，结果壳每秒 `Engine::poll_glosses` 经 `Translator::learn` 写进 `qingjian-translate::PersonalGlossary`
  （`user-glossary-<语言>.tsv`，`LayeredTranslator` 个人表优先）；随云联想开关一起开。
- 问字键（缺省 `u`）开头是问字模式（`PredictionKind::Question`，答案带读音、不校验拼音），`?` 开头要 `ModeKeys::question_mark` 开着才算（配置 `[shortcut] question_mark`，缺省关，壳用 `Engine::takes_question_mark` 决定空缓冲区的 `?` 是入口还是标点）；`PredictionKind::Translate` 是壳里快捷键触发的「翻译选中文字」
  （双向：汉字为主译成学习语言，外文译回中文，`prediction::translation_target`），译文走结果的 `sentence`。

## crates/qingjian-format

`.qj` 数据容器（`Container` mmap 读、`Writer` 写、`Table<T>` / `Text` 零拷贝视图、`hash` 可落盘哈希索引、`Metadata` 名称 / 许可证 / 署名）。
词库与语言模型都能 `write_qj` / 从 `.qj` 打开，启动 50 ms；`cargo run --release -p qingjian-dict-convert -- pack dict|lm --name … --license …`
生成 `data/generated/{dict,lm}.qj`，`bundle.sh` 在 TSV 更新时自动重打并只把 `.qj` 打进包。设计见 `docs/design/architecture.md`「数据文件：`.qj` 容器」。

## crates/qingjian-qwen

`QwenScorer`，Core `sentence::SentenceScorer` trait 的另一个实现：llama.cpp（`llama-cpp-2` 0.1.156，feature `runtime` 门控，
不开是空壳、默认构建与 CI 不拉 C++ 依赖）加载 Qwen3.5-2B 的 GGUF（`data/model/Qwen3.5-2B-UD-Q4_K_XL-text.gguf`，1.1 GB，gitignore，词表已裁剪），
给「前文 + 整句」按 BPE token 累加 log 概率，Metal 加速，双向上下文与修正建议（`max_adjustment` 30 nat）。
每条候选拼上前文各自成一个序列、一次 decode 打一批（超过 31 条或 480 token 切批），打完 `clear_kv_cache` 整体重算——
Qwen3.5 是注意力 + SSM 混合架构，`seq_cp` / 中间回卷都不可用（坑与调参记录见 `docs/notes/qwen-rescoring.md`）。
**跨长度比神经分要归一**：log 概率逐字累加，长短不一的候选直接比会让长候选永远吃亏（深池词按每字平均分、纠错变体与等长基线比，见 `split_local_scores`）。
空前文退 BOS/EOS（qwen35 没设 `dec_start_token_id`，是 -1）。壳的装配只认 GGUF（用户目录 > 包内,`apps/macos` 的 `host/model` 与 `paths::qwen_path`）；
CLI `--qwen <gguf>`（要 `--features qwen` 编译,参数族 `--neural-*`）、本地联想 `--local-prediction`（要配 `--neural-async`，
整句评测报告加「本地联想/纠错」一行）。`RescoreWorker` 后台线程在 `Drop` 里先关任务通道再 `join`——
不等它退完就析构，进程退出时 ggml 会撞 Metal 资源集断言。Engine 侧的新缺省:`NEURAL_GATE` = 2（个人证据保护闸）、
`RESCORE_CONTEXT_CHARS` = 200：光标前后的**总预算**，按 `split_window` 自适应分（两侧都有对半 100/100，
只有一侧全给、短侧用不完的让给对侧）；前文按句界结构化装填、后文取头部（`rescoring_window`）。一两个音节的输入由词级候选决定，不唤醒 2B，避免单词去重前的无效推理，见 `docs/notes/qwen-rescoring.md`。
Viterbi 侧先取 16 条并按分歧位置组合做多样化 shortlist，最终仍只给 2B 8 条，不增加神经前向数。

## crates/qingjian-lm

`BigramModel`，Core `sentence::LanguageModel` trait 的实现，从 `data/generated/lm.qj`（或 `lm-unigram.tsv` / `lm-bigram.tsv`）加载
（没有这两个文件就退化为一元词频整句）。数据由 `tools/corpus/parquet_to_text.py`（uv 脚本，HF parquet → 简体纯文本）加
`cargo run --release -p qingjian-dict-convert -- bigram --phrases assets/lexicon/phrases.tsv --phrases assets/lexicon/domain_words.tsv --brand assets/lexicon/brand.tsv --brand assets/lexicon/mixed_words.tsv data/corpus/*.txt` 生成；语料在 `data/corpus/`（gitignore）。
短语层不当 token 统计（分词时摘掉、统计完按成分合成一元 / 二元，短语得分等于原来两个词的路径，见 `bigram.rs` 模块注释），品牌词按给定次数写进一元与句首二元。
`LanguageModel::successors` 枚举前词的后继（CSR 段切片按计数排序截断），给本地联想出接续提议；trait 缺省空实现，个人 n-gram（`UserNgram::successors`）同名方法与它合并。

## crates/qingjian-platform

`Config`（TOML 配置文件，`[general]` / `[shortcut]` / `[fuzzy]` / `[dictionaries]` / `[apps]` / `[predict]` 分节，首次运行写模板，
`set_value` 用 toml_edit 原地改键保留注释、`remove_value` 原地删键恢复缺省；`[model]` 本地整句模型（`enabled` 开关、`max_adjustment` 重排幅度上限，缺省跟随模型文件建议），`LocalModelConfig`）；
`extra_dictionaries` 列出 / 加载随包领域词库与用户 `dicts/`（同名 `.qj` 优先于 `.tsv`）。

## crates/qingjian-render

自绘渲染器：候选窗一帧 + 主题 → 预乘 RGBA 位图，tiny-skia 栅格 + cosmic-text 文字（fontdb 按平台清单只加载几个字体文件、不扫系统），
自己解析 `trak` 字距表、按主题 gamma 加深笔画；cosmic-text 打了 `opsz` 光学字号补丁（qingjian-team/cosmic-text 分支 `qingjian-opsz`，workspace `[patch.crates-io]` 钉 rev）。
`examples/preview.rs` 出 PNG 与真机截图并排比、`--measure` 与 AppKit 对宽度。mac 壳 `candidates/bitmap/` 贴位图，`[general] renderer = "system"` 切回 AppKit 绘制
（过渡期退路，偏好设置「候选窗口」页可选）；`[general] font` 是候选窗字族名（空为系统字体，`bitmap/font_files.rs` 用 CoreText 按字族名找文件只加载那几个，没装就回系统字体；
设置页 `preferences/font_picker/` 是搜索框 + 列表）。设计与验收见 `docs/design/rendering.md`。

## apps/cli

测试工具，`cargo run -p qingjian-cli -- kaifa`。

- `--predict` 强制开云联想并等结果打印，交互模式下上屏后也联想。
- `--typing` 逐键计时（性能测试用 release 构建跑，目标每键 10 ms 以内）。
- `--chinese-first` 打开中文优先（`[general] chinese_first = true` 的排法），配合 `--replay` 比两种英文词位置。
- `--replay <input-log.jsonl>` 回放评测：把日志里每次上屏的键重新喂给引擎，按来源算首选 / 前五命中率、平均名次、不在候选的条数，打印没命中的例子（`--misses N`）；
  只在内存里学习不写文件，加 `--user-dict` 可带上现有学习数据。
- `--tune 名=值`（逗号分隔）覆盖个人 n-gram 插值与敲错代价的常数扫网格（名字见 `apps/cli/src/tuning.rs`，Core 侧是 `Engine::set_interpolation` / `set_typo_costs`，壳只用缺省值）。
- `--eval-text <文本>...` 整句评测：把用户自己写的中文文本按标点切句、按词库读音转成全拼，冷启动喂给引擎看整句能不能还原原句
  （首选命中率 / 字准确率 / 查询耗时；不依赖日志里当时选了什么，给整句排序与语言模型的改动当尺子），`--eval-save` 冻结成 `句子\t拼音\t上文` 三列文件，
  之后直接 `--eval-text` 它保证比的是同一份句子（本机的在 `data/eval/sentences.tsv`）。排序、整句、纠错的改动先跑它们再合。

## apps/macos

IMK 输入法，源码按 `app / host / imk / candidates / menubar / preferences` 分目录。

- 输入法菜单（状态项 + 系统输入源菜单）与偏好设置窗口都是配置文件的前端：只写 `config.toml`，`Host::apply_config` 一条通路热加载，激活期间每秒看一次文件 mtime。
- `apps/macos/scripts/bundle.sh --install` 打包安装到 `~/Library/Input Methods/`（开发用），`--pkg` 做分发用的 pkg（装 `/Library/Input Methods/`，postinstall 跑 `qingjian-macos --register`
  安装结束后由用户 launchd 的一次性任务刷新 `TextInputMenuAgent` / `imklaunchagent`，再跑一次 `qingjian-macos --register` 注册、启用并切成当前输入源；启用请求只发一次并等待用户授权，避免系统权限框循环弹出；
  签名 / 公证靠 `QINGJIAN_SIGN_IDENTITY` / `QINGJIAN_INSTALLER_IDENTITY` / `QINGJIAN_NOTARY_PROFILE`，没设就 ad-hoc；`QINGJIAN_TARGET` 指定架构，
  成品 `target/pkg/Qingjian-<版本>-<arm64|x86_64>.pkg`）；`scripts/uninstall.sh` 卸载。
- 日志在 `~/Library/Logs/Qingjian/`（按天分文件留 7 天，删了会重建），用户数据与配置在 `~/Library/Application Support/Qingjian/`。
- 配置项：云联想 `[predict]`（偏好设置「云服务」页有「测试连接」按钮：`qingjian_predict::ConnectionTest` 起线程发一条最小请求，`Host` 用独立定时器 `CloudTestMonitor` 轮询结果显示到窗口底部；
  `reasoning_effort` 缺省 `none`，DeepSeek V4 默认思考，不关正文为空）；模糊音 `[fuzzy]` 默认都关；`[general]` 学习语言（`off` 不显示译文）/ 每页候选数 / 翻页键 / 外观 / 竖排横排 / 拼音显示位置 /
  英文模式候选开关 / 中文优先 `chinese_first` / 双拼方案 `shuangpin`（小鹤 / 自然码 / 微软 / 搜狗，空为全拼）/ 日志级别 `log_level`（缺省 info 不含敲的内容，debug 逐键记，热切换）/ 输入日志 `input_log`；
  `[shortcut]` 模式键 v / u、`question_mark`（缺省关，开了空缓冲区敲 `?` 进问字）、上屏第一 / 第二个译词的修饰键 `translation` / `translation_second`、删候选 `delete_candidate`（缺省 shift，用户词整删、词库词清学习）、翻译选中文字 `translate_selection`；
  `[apps] english_candidates_off` 按 bundle identifier 列出英文模式不给候选的应用（缺省终端 / 编辑器 / IDE，`*` 前缀匹配）；
  `[dictionaries] domains` 打开随包的领域词库（`Resources/dicts/` 11 本，缺省只开 `idioms`），`disabled` 关掉用户目录 `dicts/` 里的某本导入词库；
  偏好设置「词库」页随包的可开关、导入的可开关 / 移除，可导入 TSV / Rime yaml / .qj。
- 系统文本替换（系统设置「键盘 → 文本替换」）：`host/config/text_replacements.rs` 从 `NSUserDefaults` 全局域读 `NSUserDictionaryReplacementItems`
  （每条 `{ on, replace, with }`），激活输入法时重读，变了就经 Core `merge_replacements` 并进配置里的自定义短语再 `set_custom_phrases`；
  `[general] system_text_replacements` 开关（缺省开，「自定义短语」页勾选框），内容可能含证件号、地址，日志只记条数。
- 输入法进程由 launchd 拉起，看不到 shell 的环境变量：密钥写进配置同目录的 `.env`（`QINGJIAN_API_KEY=...`，输入法启动时 dotenvy 读入）或 `config.toml` 的 `api_key`。
- 本地整句模型：`bundle.sh` 把 `data/model/`（或 `QINGJIAN_MODEL_DIR`）三件套打进 `Resources/model/`，用户目录 `model/` 优先；`host/model/mod.rs` 在后台线程加载并预热（首次 Metal 编译）后
  `set_async_sentence_scorer` 接上，`refresh` 每键先读应用光标前 64 字给 Engine 当前文、查询后 `schedule_rescoring`，`RescoreMonitor` 停键 80 ms 请求、20 ms 轮询，
  结果到了重查一次只重画当前页（翻过页 / 动过高亮不动）；「云服务」页有开关（`[model] enabled`）。
- 本地整句联想（`[predict] local`，缺省开，要与 `[model] enabled` 同时开）：Core `engine/prediction` 里实现，与云联想共用请求 / 序号 / Tab 接受 / 词槽管道。
  三类提议：①接续提议 = 静态 LM 与个人 n-gram 的后继合并（`LanguageModel::successors` + `UserNgram::successors`，个人一次计数折 8 次静态），
  胜过基线（不带续写的 guess）才出整句；②深池词 = 候选列表第一页（9 条）之外被静态排序埋没的词，
  与参照比「每字平均分」（log 概率逐字累加，不归一的活长候选天然吃亏）才榜进词槽（条数随 `[predict] slots`）；
  ③纠错变体 = 最优转换路径（2–6 个词）的每个词换同音词（`lookup_exact_all`），静态整句分预筛后取前 24 条；
  变体与 guess 等长，与**基线 guess** 比（不是与长短不一的参照比），高出 `CORRECTION_MARGIN`(2.0 nat) 才榜进词槽；
  本机给的联想词走 `CandidateKind::Local`（行为同 `Cloud`，候选窗**不画云朵**，免得误以为数据外传）。
  2B 只排序不生成：`RescoreWorker` 新增 `submit_predict` 任务类型，重排优先、同类只算最新，结果分渠道（`poll` / `poll_predict`）互不抢。
  壳侧有 250ms 防抖（`PredictMonitor::schedule` → `Host::start_prediction`），一次预测批次约 30 条——
  此前每键一发、批次上百条，实测把重排挤到超时（`texts=408 ms=5147`）。
  门控：音节 < 3 不发（与短词门控同线）；上下文前后各 100 字（`Engine::prediction_context_window`，壳按它读应用文本）；打分分段与阈值在 `split_local_scores`。
- 端到端验证可用 `osascript` 的 System Events 往 TextEdit 发按键再读回文本（终端需要辅助功能权限；输入法得在中文模式）。

## assets

- `assets/sample/`：手写样例词库与释义表，不是产品数据。
- `assets/emoji/emoji-zh.tsv` / `emoji-en.tsv`：Unicode CLDR 中文 / 英文 annotations 转出的 emoji 表（Unicode License v3，可发布；中文词与英文词各配 emoji，两张表加载时合成一张），
  `cargo run --release -p qingjian-dict-convert -- --out-dir assets/emoji emoji --language zh data/cldr/annotations-zh.json data/cldr/annotationsDerived-zh.json`（en 同理）。
- 英文词表词频：`uv run tools/corpus/english_frequency.py data/generated/english.tsv -o data/generated/english-frequency.tsv`，再 `... english <词表> --frequency <那个文件>`。

## tools/gloss-gen

用 LLM 批量生成释义表：`cargo run --release -p qingjian-gloss-gen -- generate`（密钥读 `QINGJIAN_API_KEY`，结果 JSONL 在 `data/generated/`，不进 git、可续跑，`--limit 80` 试跑）
再 `... export`（写 `glossary-{en,ja}.tsv`，产品数据在 `assets/glossary/`，见那里的 README；格式 `词\t词性. 译词[|假名]`）。CLI 与 bundle.sh 用的就是这两个文件。

## tools/dict-convert

产品数据的生成工具，输出到 `data/generated/`（gitignore）。

- `lexicon`：从 `assets/lexicon/`（自建词库源：规范字 + 常用词 + THUOCL 领域词）加 Unihan 读音（`data/unihan/Unihan_Readings.txt`）、LLM 多音字标注（`gloss-gen pinyin`，
  结果 `data/generated/pinyin-llm.jsonl`，不进 git）、语料词频（`lm-unigram.tsv`）建基础词库 `dict.tsv`（8.7 万条），并把 THUOCL 领域词按语料次数 < 50 拆成
  `dicts/<领域>.tsv` + `.qj`（11 本、13 万条，`--domain-keep-min`），流程见 `assets/lexicon/QINGJIAN.md`；`--extra-words` 并入人工挑的领域词 `assets/lexicon/domain_words.tsv`。
- `english`：转 `assets/lexicon/05_english/00_all_words.tsv`；`cedict`：释义表备用来源。
- `bigram`：统计语料；`--phrases` 给短语层、`--brand` 给品牌词（`assets/lexicon/brand.tsv`，青简 210）与中英混杂词（`mixed_words.tsv`，C盘 / B站：合成计数要成分词在语料里，C 不是 token，只能直接给一元，次数对着同音竞争词定），领域词也走合成计数（语料里只有几十次的词当 token 统计会吸走成分词的二元证据）。
- `mine`：从语料挖词库没收的高频词并过滤（`oov_filter.rs`：虚词规则 + 相邻字对 PMI≥3，`--candidates` 只重过滤）。
- `phrases`：挖短语层（两遍扫语料：相邻两词、两段二元都够频的相邻三词，总次数与对话语料次数都 ≥ 2000 + 边界规则，读音由成分词拼出；我的 / 不知道 / 有没有 这类常用词表不收的组合，
  `assets/lexicon/phrases.tsv`；词库已并入过短语时重跑加 `--refresh`）。
- `pack dict|lm|glossary`：打 `.qj`（释义表也进容器）。
