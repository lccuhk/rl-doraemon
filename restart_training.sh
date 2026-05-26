#!/bin/bash

echo "========================================"
echo "麻将AI训练 - 重启脚本"
echo "========================================"

cd "$(dirname "$0")"

echo ""
echo "步骤 1: 停止当前训练进程..."

CURRENT_PID=$(pgrep -f "train_2000_episodes.py" | head -1)

if [ -n "$CURRENT_PID" ]; then
    echo "发现运行中的训练进程 (PID: $CURRENT_PID)"
    echo "发送 SIGTERM 信号，优雅退出..."
    kill -SIGTERM "$CURRENT_PID"
    
    echo "等待进程退出..."
    sleep 5
    
    if pgrep -f "train_2000_episodes.py" > /dev/null; then
        echo "进程仍在运行，强制终止..."
        kill -9 "$CURRENT_PID"
        sleep 2
    fi
    
    echo "进程已停止"
else
    echo "未发现运行中的训练进程"
fi

echo ""
echo "步骤 2: 检查可用的检查点..."

CHECKPOINT_DIR="./checkpoints_2000"
LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_*_agent0.pt 2>/dev/null | head -1)

if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "找到最新检查点: $LATEST_CHECKPOINT"
    
    EPISODE=$(basename "$LATEST_CHECKPOINT" | cut -d'_' -f2)
    NEXT_EPISODE=$((EPISODE + 1))
    
    echo "检查点对应回合: $EPISODE"
    echo "将从第 $NEXT_EPISODE 回合继续训练"
else
    echo "未找到检查点，将从头开始训练"
fi

echo ""
echo "步骤 3: 启动新的训练进程..."
echo "========================================"
echo "训练配置:"
echo "  - 总回合数: 2000"
echo "  - 每回合自动保存: 是"
echo "  - 保留快照数: 10"
echo "  - 信号控制:"
echo "    * 手动保存: kill -SIGUSR1 <PID>"
echo "    * 优雅退出: kill -SIGTERM <PID>"
echo "========================================"
echo ""

if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "从检查点恢复训练..."
    nohup python3 train_2000_episodes.py --resume > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
else
    echo "从头开始训练..."
    nohup python3 train_2000_episodes.py > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
fi

sleep 2

NEW_PID=$(pgrep -f "train_2000_episodes.py" | head -1)

if [ -n "$NEW_PID" ]; then
    echo ""
    echo "========================================"
    echo "训练已启动！"
    echo "进程ID: $NEW_PID"
    echo ""
    echo "常用命令:"
    echo "  查看日志: tail -f training_*.log"
    echo "  手动保存: kill -SIGUSR1 $NEW_PID"
    echo "  优雅退出: kill -SIGTERM $NEW_PID"
    echo "  查看进程: ps aux | grep train_2000"
    echo "========================================"
else
    echo ""
    echo "错误: 训练进程启动失败"
    echo "请检查日志文件获取详细信息"
fi
