#!/bin/sh
# 安装器的 postinstall 早于 PackageKit 最后的 bundle 登记；稍后在登录用户会话里刷新输入法代理。
set -u
sleep 5
pkill -x TextInputMenuAgent 2>/dev/null || true
pkill -x imklaunchagent 2>/dev/null || true
exec "/Library/Input Methods/Qingjian.app/Contents/MacOS/qingjian-macos" --register
