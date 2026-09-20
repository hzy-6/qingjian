# 发版流程

2026-09-07 搭起来的：GitHub Actions 按标签打包、建 Release、生成官网下载页用的 `releases.json`。
这里记怎么发一版、各环节的依赖，以及官网怎么消费产物。

## 一次发版做什么

1. 改 `apps/macos/Cargo.toml` 的 `version`（`apps/macos` 的 Info.plist 版本号从这里取，pkg 文件名也是）：把 `0.1.2-dev` 改成 `0.1.2`。
   **发版之间版本号一直带 `-dev`**（Rust nightly / Firefox Nightly 那套）：Cargo.toml 写 `0.1.2-dev`，`bundle.sh` 打包时再接上 git 短哈希，
   本地装的、CI 中间构建的都显示 `0.1.2-dev-1a2b3c4`（工作区有改动加 `+`），测试时一眼知道装的是哪个提交；版本号干净的一定是线上包；
   带 `-dev` 的标签 CI 直接拒绝。pkg 的 `--version` 与 `distribution.xml` 只认数字点号，`bundle.sh` 去掉后缀再传，Info.plist 与 pkg 文件名保留完整版本。
2. `CHANGELOG.md` 顶上加一节 `## <版本> · <日期> · <渠道>`（渠道是 `alpha` / `beta` / `rc` / `stable`），一行一条、面向用户的措辞。
   **更新日志手写，不由提交自动生成**：提交信息里有大量内部改动（拆模块、修 RefCell 重入），用户看不懂也不关心；
   做法是发版前按上个标签以来的 `git log` 起草几条，人审一遍再定稿。
3. 提交，打**带平台前缀**的注释标签并推：`git tag -a macos-v0.1.1 -m "青简 macOS 0.1.1" && git push origin main macos-v0.1.1`
   （标签使用 `macos-v*` 前缀）。
3b. 标签推出去之后紧接一个普通提交把版本号改成下一个开发版（只是改 Cargo.toml，不打标签、不建 Release；-dev 版本永远没有标签与 Release）：`apps/macos/Cargo.toml` 改成 `0.1.3-dev`，本地从此打的包都带 `-dev`。
4. `release.yml` 跑完后 GitHub Release 上有 `Qingjian-<版本>-arm64.pkg`、`Qingjian-<版本>-x86_64.pkg`、`SHA256SUMS`、`build-info.json`（提交、构建时间、工具链）、`releases.json`。
5. 官网由 Cloudflare Workers Builds 按官网仓库的提交自动构建，没有可调用的构建钩子，所以主仓库靠**往官网仓库推一个小提交**来触发：
   `tools/release/bump-website.sh` 把版本标签与文档提交号写进官网的 `src/content/upstream.json` 并提交推送（提交者 qingjian-ci）。
   配了 `QINGJIAN_WEB_TOKEN`（对 qingjian-web 有 Contents: read and write 的 fine-grained PAT）release.yml 末尾自动做；
   官网文档只随发版更新（`docs/user/` 平时改动不推官网，免得文档领先于用户装到的版本）。没配就在官网仓库随便提交一次（或本地跑这个脚本）。
   官网构建时才拉最新 Release 的 `releases.json` 与主仓库 `docs/user`，所以提交内容本身不重要，`upstream.json` 只是留个记录、
   顺便让文档按记下的提交号拉（版本对得上）。

workflow 会核对 `apps/macos/Cargo.toml` 版本号与标签（去掉 `macos-v` 前缀后）一致，不一致直接失败，避免打出版本号错的包。

Rust 工具链由 `rust-toolchain.toml` 钉版本（现在 1.96.0），两个 workflow 里 `dtolnay/rust-toolchain@master` 的 `toolchain:` 输入写同一个号；升级 Rust 时三处一起改。

## 提交前检查与 CI

本地 `git config core.hooksPath .githooks` 启用一次后，每次提交前 `.githooks/pre-commit` 先拒绝装饰性分隔注释（`// ====` / `// ────`，只做视觉分组不带「为什么」），再跑 `cargo fmt --check` 与 `cargo clippy -D warnings`（含 IMK 外壳，增量几十秒）；
`.githooks/pre-push` 在推之前跑全 workspace 测试。外部 PR 走同一套 `ci.yml`，不过不合。

供应链：workflow 里的 actions 一律钉到 commit（注释写对应标签），`.github/dependabot.yml` 每周一提 Cargo 与 actions 的更新 PR；`audit.yml` 每周与 Cargo.lock 变动时跑 `cargo audit`；
cargo 命令全 `--locked`（含 `bundle.sh`）。普通 CI 只有 `contents: read`，checkout 不留凭据；release 的 secrets 不放顶层 env，只注入用它的那一步。

发版门禁（`release.yml` 第一步）：版本号与标签一致且不带 `-dev`；标签指向的提交必须在 `main` 上（`git merge-base --is-ancestor`）；产品数据下载后按 `data` Release 的 `SHA256SUMS` 校验，摘要写进 `build-info.json` 的 `data_sha256`。
**正式版前还欠**：产品数据改成不可变 tag 并在仓库里锁定版本（现在滚动覆盖，同一源码 tag 重跑可能拿到不同数据）、安装包内容验证（词库 / 模型 / 许可齐不齐、签名校验）。

## Workflow

| 文件 | 触发 | 做什么 |
|---|---|---|
| `.github/workflows/ci.yml` | push main、PR | Linux 检查核心 crate，macOS 检查 IMK 壳 |
| `.github/workflows/release.yml` | 推 `macos-v*` 标签 | 下载产品数据、打包、可选签名公证、建立 Release |

## 产品数据从哪来

词库、语言模型、释义表（`data/generated/*.qj`、`dicts/*.qj`、英文词表）不在 git 里，体积约 90 MB 且由本机数据管道生成。
它们发在仓库里一个个**不可变**的预发布 Release 上：`data-v1`、`data-v2`……每次数据重生成发一个新号、从不覆盖
（预发布不会成为 GitHub 的 latest，官网取 latest 时不会拿到它）。仓库里 `tools/release/data.lock` 钉住当前要用的标签与两个资产的 SHA-256，
跟用到新数据的代码同一个提交进去：checkout 哪个提交就拿到它对应的那版数据，离线自编译的人不会因为我们改了数据而编出坏包。

- `tools/release/data-bundle.sh`：把 `data/generated/` 打成 `qingjian-data.tar.gz`，本地整句模型单文件 `data/model/模型 GGUF`
  （训练仓库导出GGUF到 `data/model/`，`tools/release/data-bundle.sh(模型随数据链分发)` 打成一个 `.qj` 容器，fp16 约 56 MB，元数据也写在那个脚本里）原样上传，
  连同 LLM 续跑中间产物 `qingjian-llm-intermediates.tar.gz` 发到下一个 `data-vN`（`--tag` 可指定，已存在就拒绝），然后改写 `data.lock`。
- `tools/release/data-fetch.sh`：按 `data.lock` 下载（有 gh 用 gh，没有就 curl 直连）、按锁文件里的哈希校验（不信 Release 自己那份 `SHA256SUMS`），
  数据包解到 `data/generated/`、`模型 GGUF` 放到 `data/model/`。`release.yml` 和离线自编译走同一个脚本；
  `bundle.sh` 见到 `dict.qj` 就按产品数据打包、见到 `模型 GGUF` 就放进 `Resources/model/`。
  标签与两个哈希记进 `build-info.json`（`data_tag` / `data_sha256` / `model_sha256`）。

数据重生成之后（重跑 lexicon / bigram / gloss-gen export）或模型重训之后跑一次 `data-bundle.sh`（GGUF比 `.qjm` 新会自动重打），
把锁文件的改动提交（`chore(data): 数据 data-vN`），否则 CI 打的包还是锁文件指的旧数据。模型文件缺失或哈希不符时 CI 会失败，不会静默地发出错数据的包。
2026-09-16 之前用的是滚动覆盖的 `data` Release，已冻结不再更新。

## 签名与公证

没有证书时 CI 照样出包（ad-hoc 签名，Release 说明里自动加一句「首次打开要在隐私与安全性里放行」）。
Apple Developer 账号有了以后，在仓库 Secrets 里配齐 `release.yml` 头部注释列的七个值（.p12 与 .p8 都 base64），
下一次发版就是签名 + 公证 + 钉票据的包，用户下载双击即装。`bundle.sh` 本身通过 `QINGJIAN_SIGN_IDENTITY` /
`QINGJIAN_INSTALLER_IDENTITY` / `QINGJIAN_NOTARY_PROFILE` 三个环境变量工作，本机有证书也能这样打。

## releases.json：官网下载页的数据源

`tools/release/releases_json.py` 从 `CHANGELOG.md`（日期、渠道、更新日志）、GitHub Releases API（附件、地址、大小）
与每次发布的 `SHA256SUMS` / `build-info.json`（每个包的 sha256、提交哈希、构建时间、工具链）生成，挂在每个版本的 Release 上；官网固定取
`https://github.com/<repo>/releases/latest/download/releases.json`（仓库私有期间要带令牌走 API 下载附件）。

结构对应官网 `src/lib/releases.ts` 里的 `Release` / `Asset` 类型：

```json
{
  "generated": "2026-09-07T12:00:00Z",
  "repository": "owner/qingjian",
  "latest": "0.1.0",
  "releases": [
    {
      "version": "0.1.0",
      "date": "2026-09-07",
      "channel": "beta",
      "notes": ["整句输入：……", "候选旁有词性和译词……"],
      "commit": "869ad00…（40 位）",
      "built_at": "2026-09-07T08:38:12Z",
      "toolchain": "rustc 1.96.0 (ac68faa20 2026-05-25)",
      "assets": [
        { "platform": "macos", "arch": "Apple Silicon", "file": "Qingjian-0.1.0-arm64.pkg",
          "url": "https://github.com/owner/qingjian/releases/download/v0.1.0/Qingjian-0.1.0-arm64.pkg",
          "size": 35989277, "sha256": "…" },
        { "platform": "macos", "arch": "Intel", "file": "Qingjian-0.1.0-x86_64.pkg", "url": "…", "size": 36172871, "sha256": "…" }
      ]
    }
  ]
}
```

- `releases` 从新到旧，`latest` 是第一条的版本号；官网「当前版本」取它，历史版本列表就是整个数组。
- `channel` 是 `alpha` / `beta` / `rc` / `stable`，显示成什么字由官网定；`commit` / `built_at` / `sha256` 给用户核对下载的包，下载页应显示 sha256 与提交短哈希。
- `SHA256SUMS` 与 `releases.json` 自己不列进 `assets`。
- 官网侧要做的：构建时下载这个文件替代手写的 `releases` 数组（与拉 `docs/user` 的 `sync-docs.mjs` 同一处、同一个令牌），
  `downloadsOpen` 开关仍由官网自己控制。

## 本机打包

`apps/macos/scripts/bundle.sh --pkg` 打本机架构；`QINGJIAN_TARGET=x86_64-apple-darwin` 交叉编译 Intel 包（要先 `rustup target add`，
本机不需要时不必装，CI 上两个都打）。成品在 `target/pkg/Qingjian-<版本>-<arch>.pkg`，每个架构一个工作目录，连着打互不覆盖。
