#!/bin/bash

# 设置日志目录
LOG_DIR="$HOME/pytorch_build_logs"
mkdir -p "$LOG_DIR"

# 生成时间戳
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/build_$TIMESTAMP.log"

# 打印开始信息
echo "=========================================="
echo "  PyTorch Build Started: $TIMESTAMP"
echo "  Log file: $LOG_FILE"
echo "=========================================="

# # 设置必要的环境变量（如需要）
# export PATH=$HOME/pytorch/third_party/sleef/bin:$PATH

# 可选：清理旧的构建
echo "[*] Cleaning previous build..." | tee -a "$LOG_FILE"
rm -rf build
python3 setup.py clean >> "$LOG_FILE" 2>&1

# 正式构建
echo "[*] Running setup.py develop --cmake..." | tee -a "$LOG_FILE"
python3 setup.py develop --cmake >> "$LOG_FILE" 2>&1

# 输出构建结果状态
if [ $? -eq 0 ]; then
    echo "[✓] Build succeeded at $TIMESTAMP" | tee -a "$LOG_FILE"
else
    echo "[✗] Build failed at $TIMESTAMP" | tee -a "$LOG_FILE"
    echo "[!] Check full log at: $LOG_FILE"
fi
