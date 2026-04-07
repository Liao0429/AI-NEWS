#!/bin/bash

# 运行所有基准测试的脚本
echo "开始运行所有基准测试..."
echo "="
echo ""

# 设置Python路径
export PYTHONPATH="."

# 运行所有资产的实验
echo "1. 运行所有资产的实验..."
echo ""
python main_runner.py --all-assets

# 运行单个资产的详细分析
echo ""
echo "2. 运行单个资产的详细分析..."
echo ""
for asset in AAPL MSFT GOOGL "BTC-USD"
do
    echo "运行资产: $asset"
    python main_runner.py --asset "$asset"
    echo ""
done

echo "3. 生成综合报告..."
echo ""
# 这里可以添加生成综合报告的命令

# 清理缓存（可选）
echo "4. 清理缓存..."
echo ""
rm -rf data/cache/*

echo "="
echo "所有基准测试运行完成！"
echo "结果保存在 results/ 目录中"
