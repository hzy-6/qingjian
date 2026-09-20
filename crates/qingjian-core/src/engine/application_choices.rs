//! 按应用隔离的候选偏好：全局记录保留跨应用泛化，应用内记录提供更强的局部信号。

use super::{Engine, Learner};

const APPLICATION_CHOICE_WEIGHT: i32 = 2;
const APPLICATION_KEY_PREFIX: &str = "\u{1f}app:";

fn application_key(application: &str, input: &str) -> String {
    format!("{APPLICATION_KEY_PREFIX}{application}\u{1f}{input}")
}

impl Engine {
    pub(super) fn record_scoped_choice(&mut self, input: &str, text: &str) {
        self.learner.record_choice(input, text);
        if let Some(application) = self.application.as_deref() {
            self.learner
                .record_choice(&application_key(application, input), text);
        }
    }

    pub(super) fn unrecord_scoped_choice(
        &mut self,
        application: Option<&str>,
        input: &str,
        text: &str,
    ) {
        self.learner.unrecord_choice(input, text);
        if let Some(application) = application {
            self.learner
                .unrecord_choice(&application_key(application, input), text);
        }
    }

    pub(super) fn record_scoped_negative(
        &mut self,
        application: Option<&str>,
        input: &str,
        text: &str,
    ) {
        self.learner.record_negative(input, text);
        if let Some(application) = application {
            self.learner
                .record_negative(&application_key(application, input), text);
        }
    }

    pub(super) fn scoped_choice_balance(&self, input: &str, text: &str) -> i32 {
        let global = self.learner.choice_balance(input, text);
        let application = self.application.as_deref().map_or(0, |application| {
            self.learner
                .choice_balance(&application_key(application, input), text)
        });
        global.saturating_add(application.saturating_mul(APPLICATION_CHOICE_WEIGHT))
    }
}
