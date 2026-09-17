# macOS 候选窗自绘渲染器

## 设计

候选窗口由 `qingjian-render` 负责布局和绘制，输入为候选、拼音行、译文、分页提示与主题，输出位图。macOS 壳使用 NSPanel 显示位图；偏好设置与菜单继续使用 AppKit 原生控件。

渲染使用 `tiny-skia` 栅格化、`cosmic-text` 排字及回退字体。字体按需加载，主题控制颜色、字体、圆角与阴影。`[general] renderer = "system"` 可切回 AppKit 绘制。

## spike 结果（2026-09-13 晚，macOS）

代码：`crates/qingjian-render`（渲染器）、`examples/preview.rs`（离线出 PNG + 量宽度）、`apps/macos/src/candidates/bitmap/`（壳侧贴位图；`[general] renderer = "system"` 切回 AppKit 绘制，偏好设置「候选窗口」页可选，是过渡期退路，稳定一个版本后删）。
对比方法：TextEdit 里敲 `nihao`，`screencapture -l` 抓真实候选窗；同一帧人工抄进样例，渲染成 PNG 并排；再把渲染器装进壳抓真机。

| 验收 | 结果 |
| --- | --- |
| 彩色 emoji | 过。Apple Color Emoji（sbix）经 swash 出 RGBA 位图，👋 🙂 与原生一致 |
| 中日字形回退 | 过。locale `zh-CN` 下汉字全部落到 PingFang SC（含「日本語」「骨直曜」），没拿 Hiragino 画中文；假名落到 PingFang HK |
| 字体按需加载 | 过。5 个文件 67 张面，release 1.5–4 ms（mmap，只解析名字表与 cmap）；预览工具整进程峰值 RSS 16 MB |
| 文字观感 | 过（mac）。补了三样才对齐，见下 |

首帧：2 倍屏 266×300 pt 一帧 release 1 ms（首帧 6 ms，含字形栅格缓存冷启动）。壳内字体库 3.8 ms。

**对齐原生要补的三样**（都是 CoreText 对系统字体默默做的事）：

1. **光学字号 `opsz`**：SF 是变量字体，CoreText 在 20 pt 以下用 opsz = 17（Text 视觉尺寸），cosmic-text 只设 `wght`，落在缺省的 28（Display），
   小字号英文窄 16–24%。补法：给 cosmic-text 打补丁 `FontSystem::set_optical_size`（shaper 位置与 swash 栅格都带 opsz），放在 [qingjian-team/cosmic-text](https://github.com/qingjian-team/cosmic-text) 的 `qingjian-opsz` 分支（基于 0.19.0，一个提交），workspace `[patch.crates-io]` 钉 rev；上游收了就回 crates.io。
   补丁是全局一个值，字体实例缓存没按它分键；候选窗几种字号都在 20 pt 以下落同一档，够用，做主题字号可调时要改成按字号分键。
2. **`trak` 字距表**：SF 按字号给每个字形加减间距（11 pt +12、12 pt 0、16 pt −40 个字体单位，值 / upem × 字号 = 点），PingFang 没有正常轨。
   渲染器自己解析 `trak`（`fonts/trak.rs`），按字形所用字体各查各的。补完后「hello」「ni'hao」「1/6」「phr. you change」三个字号的宽度与 `NSAttributedString.size()` 到小数点后两位相等。
3. **笔画加深**：CoreText 对文字抗锯齿有一层 gamma，线性混合出来的字偏细，深色背景尤其明显。主题里加 `text_gamma`（浅色 0.85、深色 0.75），放大并排看笔画粗细一致。

**剩下的已知差异**（每个字形 ≤ 1 pt，并排看不出，记着就行）：
- 「·」（U+00B7）CoreText 在中文系统上用 `.CJKSymbolsFallbackSC` 的宽点（5.76 pt），我们用 SF 自己的（3.56 pt）；一行译文差 2 pt。
- 汉字 CoreText 用私有的 `.PingFang UI Text SC`（advance 0.993 em），我们用公开的 PingFang SC（1 em）。
- 假名 CoreText 用 `.CJKSymbolsFallbackSC`，我们用 PingFang HK。
- 云朵是矢量描边近似 SF Symbol `cloud`，比原生略粗。

真机再过了一遍：纠错后的拼音行（删除线 + 淡色剩余）、候选行里的云端词、开着候选窗切系统深浅色、配置 `renderer` 热切换两个方向；1 倍外接屏没设备没验。
假名原先落到 PingFang HK，原因是 ヒラギノ角ゴシック W3 字重 300 被 cosmic-text 的字重匹配筛掉，换 W4 后落 Hiragino Sans。
笔画加深用极性相关的覆盖率 gamma 不是我们独创：[muri #71](https://github.com/MattJackson/muri/issues/71) 得出同样结论（macOS 的字体平滑按前景 / 背景极性调），
[skip.house](https://skip.house/blog/macos-font-rendering) 提到 Patrick Walton 逆向出了 macOS 的膨胀公式（pathfinder），以后想逐像素对齐可以查它。

字体可选：`[general] font` 指定字族名，mac 壳用 CoreText 按字族名查出文件（`CTFontDescriptorCreateMatchingFontDescriptors` → `kCTFontURLAttribute`）交给渲染器只加载那几个文件，系统字体仍在后面当回退；没装就退回系统字体并记日志。真机验过 Kaiti SC 与不存在的字体名。

**结论**：四条都过，mac 上位图渲染器可以替换 AppKit 绘制。下一步做主题 TOML，稳定一版后删 AppKit 旧路径。
主题以后要放图片 / 动图 / 花边：渲染器输出就是一张位图，装饰只是多叠几层，不用换底子。
