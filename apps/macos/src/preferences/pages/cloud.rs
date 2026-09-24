//! 「云服务」页：本地整句模型开关，云联想开关、云端词格数、接口地址 / 模型 / 密钥、测试连接。

use objc2::MainThreadMarker;
use objc2::rc::Retained;
use objc2_app_kit::{NSButton, NSColor, NSPopUpButton, NSSecureTextField, NSTextField};
use objc2_foundation::NSString;
use qingjian_platform::Config;

use crate::preferences::controls::{
    button, checkbox, note, note_owned, row_checkbox, row_control, row_popup, secure_field, select,
    set_checked, text_field,
};
use crate::preferences::layout::{Layout, PAGE_PADDING, ROW_HEIGHT};
use crate::preferences::setting::{MODEL_CAPS, Setting};
use crate::preferences::target::PreferencesTarget;

/// 云端词槽位弹出菜单的上限（配置文件里可以填更大，菜单只列到这）。
const MAX_CLOUD_SLOTS: usize = 4;

/// 「模型修正上限」菜单各档的标题，与 [`MODEL_CAPS`] 一一对应。
const MODEL_CAP_TITLES: [&str; MODEL_CAPS.len()] =
    ["跟随模型", "保守（2）", "适中（4）", "宽松（8）"];

/// 字符级整句提议的说明小字：模型文件在时与不在时各一句（缺文件时提示怎么补）。
/// 两条都在布局时按较长的那条占位，切换到较短的一条时不会截断。
const CHARACTER_NOTE_OK: &str =
    "用字符 n-gram 模型给重排池补同音字级候选（会议时 → 会议室 这类）。";
const CHARACTER_NOTE_MISSING: &str = "未找到字符模型文件 char5.fst：把它放进用户数据目录的 model/（或重新打包时带上），这一项才会生效。";

/// 配置值对应菜单第几档：正好在档上用那档，手改出的中间值挑最近的一档显示（距离相同取更保守的小档；
/// 只在用户再选时才落盘）。
fn model_cap_index(value: Option<f64>) -> usize {
    let Some(value) = value else {
        return 0;
    };
    let mut nearest = 0;
    let mut best = f64::INFINITY;
    for (index, option) in MODEL_CAPS.iter().enumerate().skip(1) {
        let distance = (option.unwrap() - value).abs();
        if distance < best {
            best = distance;
            nearest = index;
        }
    }
    nearest
}

pub struct CloudPage {
    /// 本地整句模型开关。
    local_model: Retained<NSButton>,

    /// 本地整句模型的修正上限档位。
    model_cap: Retained<NSPopUpButton>,

    /// 字符级整句提议开关。
    character: Retained<NSButton>,

    /// 字符级整句提议下方的说明小字，会按模型文件在不在改文字。
    character_note: Retained<NSTextField>,

    /// 云联想开关。
    enabled: Retained<NSButton>,

    /// 云端词槽位数（0–4）。
    slots: Retained<NSPopUpButton>,

    /// 接口地址。
    base_url: Retained<NSTextField>,

    /// 模型名。
    model: Retained<NSTextField>,

    /// 密钥输入框，永远不回显已有值。
    api_key: Retained<NSSecureTextField>,

    /// 「测试连接」按钮。
    test: Retained<NSButton>,
}

impl CloudPage {
    pub fn build(layout: &mut Layout, mtm: MainThreadMarker, target: &PreferencesTarget) -> Self {
        let local_model = checkbox(mtm, "本地整句模型", Setting::LocalModelEnabled, target);
        row_checkbox(layout, &local_model);
        note(
            layout,
            mtm,
            "随包的小模型在本机给整句候选重新排序，全程离线；停键后约一百毫秒生效。关掉只用词库统计。",
        );
        let cap_titles: Vec<String> = MODEL_CAP_TITLES.iter().map(|t| (*t).to_owned()).collect();
        let model_cap = row_popup(
            layout,
            mtm,
            "模型修正上限",
            &cap_titles,
            Setting::LocalModelCap,
            target,
        );
        note(
            layout,
            mtm,
            "模型重排一次最多把一条整句候选挪多少分：调小更稳（重排幅度受限），调大更信模型。缺省跟随模型文件自带的建议。",
        );
        let character = checkbox(mtm, "字符级整句提议", Setting::LocalModelCharacter, target);
        row_checkbox(layout, &character);
        // 先按「缺文件」那条占位（较长），sync 时再换成实际提示
        let character_note = note_owned(layout, mtm, CHARACTER_NOTE_MISSING);
        let enabled = checkbox(mtm, "启用云联想", Setting::CloudEnabled, target);
        row_checkbox(layout, &enabled);
        note(
            layout,
            mtm,
            "开启后组句时会把光标附近的几十个字发给下面的服务，让模型补全整句、联想下文；密码框里绝不发送。菜单栏图标旁会带一个云朵。",
        );
        let slot_titles: Vec<String> = (0..=MAX_CLOUD_SLOTS)
            .map(|n| match n {
                0 => "不要（只要整句补全）".to_owned(),
                n => format!("{n} 格"),
            })
            .collect();
        let slots = row_popup(
            layout,
            mtm,
            "云端词位置",
            &slot_titles,
            Setting::CloudSlots,
            target,
        );
        note(
            layout,
            mtm,
            "云端词到了只补进第一页末尾这几格（比如 2 就是 8、9），前面的本地候选不动；没到就什么都不变，翻页后全是本地候选。",
        );
        let base_url = text_field(mtm, Setting::BaseUrl, target);
        row_control(layout, mtm, "接口地址", &base_url);
        let model = text_field(mtm, Setting::Model, target);
        row_control(layout, mtm, "模型", &model);
        let api_key = secure_field(mtm, Setting::ApiKey, target);
        row_control(layout, mtm, "API 密钥", &api_key);
        note(
            layout,
            mtm,
            "文本框按回车保存。密钥只保存在这台电脑上，不会随配置文件导出，也不显示已填的值。",
        );
        let test = button(mtm, "测试连接", Setting::TestCloud, target);
        layout.place(&test, PAGE_PADDING, 120.0, ROW_HEIGHT + 4.0);
        layout.next_row(ROW_HEIGHT + 4.0);
        note(
            layout,
            mtm,
            "用上面填的地址、模型、密钥发一条最小请求，结果显示在窗口底部。输入法进程看不到终端里的代理变量，走不通时先查这个。",
        );
        Self {
            local_model,
            model_cap,
            character,
            character_note,
            enabled,
            slots,
            base_url,
            model,
            api_key,
            test,
        }
    }

    /// `key_present` 是密钥已经有了（环境或配置里）；密钥框永远不回显值，只换占位文字。
    /// `model_present` 是包里或用户目录里有模型文件，没有就把本地模型的勾选与上限档位灰掉；云联想关着时它下面的项全灰。
    /// `char_present` 是字符级整句提议模型（`char5.fst`）在不在，没有就把那一项灰掉。
    pub fn sync(
        &self,
        config: &Config,
        key_present: bool,
        model_present: bool,
        char_present: bool,
    ) {
        set_checked(&self.local_model, config.model.enabled && model_present);
        self.local_model.setEnabled(model_present);
        select(
            &self.model_cap,
            Some(model_cap_index(config.model.max_adjustment)),
        );
        self.model_cap.setEnabled(model_present);
        set_checked(&self.character, config.model.character_proposals);
        self.character
            .setEnabled(model_present && char_present && config.model.enabled);
        if char_present {
            self.character_note
                .setStringValue(&NSString::from_str(CHARACTER_NOTE_OK));
            self.character_note
                .setTextColor(Some(&NSColor::secondaryLabelColor()));
        } else {
            self.character_note
                .setStringValue(&NSString::from_str(CHARACTER_NOTE_MISSING));
            self.character_note
                .setTextColor(Some(&NSColor::systemOrangeColor()));
        }
        set_checked(&self.enabled, config.predict.enabled);
        let cloud = config.predict.enabled;
        self.slots.setEnabled(cloud);
        self.base_url.setEnabled(cloud);
        self.model.setEnabled(cloud);
        self.api_key.setEnabled(cloud);
        self.test.setEnabled(cloud);
        select(&self.slots, Some(config.predict.slots.min(MAX_CLOUD_SLOTS)));
        self.base_url
            .setStringValue(&NSString::from_str(&config.predict.base_url));
        self.model
            .setStringValue(&NSString::from_str(&config.predict.model));
        self.api_key.setStringValue(&NSString::from_str(""));
        let hint = if key_present {
            "已设置，输入新值可替换"
        } else {
            "未设置"
        };
        self.api_key
            .setPlaceholderString(Some(&NSString::from_str(hint)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 档位映射：缺省与正好在档上的各归各位，中间值挑最近的、距离相同取更保守的小档。
    #[test]
    fn model_cap_index_matches_steps() {
        assert_eq!(model_cap_index(None), 0);
        assert_eq!(model_cap_index(Some(2.0)), 1);
        assert_eq!(model_cap_index(Some(4.0)), 2);
        assert_eq!(model_cap_index(Some(8.0)), 3);
        assert_eq!(model_cap_index(Some(5.5)), 2);
        assert_eq!(model_cap_index(Some(6.0)), 2);
        assert_eq!(model_cap_index(Some(1.0)), 1);
        assert_eq!(model_cap_index(Some(9.0)), 3);
    }
}
