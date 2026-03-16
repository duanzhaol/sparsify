#!/bin/bash
# LUTurbo 研究文档 - Commands 安装脚本
# 用法：在论文项目根目录下运行 bash research/setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
COMMANDS_DIR="$PROJECT_DIR/.claude/commands"

mkdir -p "$COMMANDS_DIR"

# 安装所有 .command.md 文件为符号链接
for cmd_file in "$SCRIPT_DIR"/*.command.md; do
    [ -f "$cmd_file" ] || continue
    name="$(basename "$cmd_file" .command.md).md"
    target="$COMMANDS_DIR/$name"

    if [ -L "$target" ]; then
        rm "$target"
    elif [ -f "$target" ]; then
        echo "警告：$target 已存在且不是符号链接，跳过"
        continue
    fi

    ln -s "$cmd_file" "$target"
    echo "已安装：$name → $(basename "$cmd_file")"
done

echo "完成"
