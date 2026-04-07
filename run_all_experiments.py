import sys
sys.path.append('.')

import os
import time
import numpy as np
import random
from src.experiment_logger import get_logger
from src.llm_validator import get_validator


def set_random_seed(seed: int = 42) -> None:
    """设置随机种子，确保可复现性"""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def run_robustness_test():
    """运行稳定性测试"""
    from experiments.robustness_test import run_robustness_test
    print()
    print("=" * 80)
    print("运行稳定性测试 (Robustness Test)")
    print("=" * 80)
    print()
    run_robustness_test()


def main():
    """运行所有实验"""
    print("="*80)
    print("LLM新闻分析实验系统 - 完整实验流程")
    print("="*80)
    print()
    
    # 设置随机种子
    set_random_seed(42)
    print("✓ 随机种子已设置 (seed=42)")
    print()
    
    # 初始化实验日志
    logger = get_logger()
    validator = get_validator()
    
    # 示例Prompt
    sample_prompt = """You are a financial analyst.

Given the following news, predict whether the asset price will go UP or DOWN in the next trading day.

Constraints:
- Only use the information in the news
- Do NOT assume future information
- Output strictly in JSON format:

{
  "prediction": "UP" or "DOWN",
  "confidence": 0~1,
  "reason": "brief explanation"
}
"""
    
    # 开始实验
    experiment_id = logger.start_experiment(
        model_type="llm",
        model_name="kimi-k2",
        api_version="v1",
        prompt=sample_prompt,
        temperature=0.1,
        random_seed=42,
        experiment_type="full_pipeline",
        dataset_name="From_News_to_Forecast",
        sample_count=150
    )
    
    print()
    print("="*80)
    print("提示：这个是统一入口脚本")
    print("要运行具体实验，请使用：")
    print("  - experiments/final_experiment.py")
    print("  - experiments/robustness_test.py")
    print("  - 或从 src/ 导入模块进行自定义实验")
    print("="*80)
    print()
    
    # 结束实验（示例）
    logger.end_experiment(
        results={
            "status": "demo",
            "message": "使用experiments/目录下的脚本运行具体实验"
        }
    )


from src.audit import get_auditor


if __name__ == '__main__':
    # 运行稳定性测试
    run_robustness_test()
    
    # 运行原始的主流程
    main()
    
    # 运行实验审计
    print()
    print("=" * 80)
    print("运行实验审计")
    print("=" * 80)
    auditor = get_auditor()
    auditor.run_audit()
