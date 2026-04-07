import sys
import os
import traceback
import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random

# 设置随机种子（固定，确保可复现性）
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

print("="*80)
print("LLM新闻分析实验 - 一键复现脚本")
print("="*80)
print()
print(f"随机种子: {RANDOM_SEED}")
print(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# 确保输出目录存在
os.makedirs('results/tables', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/logs', exist_ok=True)

# 日志文件
log_file = os.path.join('results', 'logs', f'reproduce_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')


def log(message):
    """记录日志到文件和控制台"""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')


def run_experiment(experiment_name, experiment_func):
    """运行单个实验"""
    log(f"开始实验: {experiment_name}")
    try:
        experiment_func()
        log(f"✓ 实验完成: {experiment_name}")
        return True
    except Exception as e:
        log(f"✗ 实验失败: {experiment_name}")
        log(f"  错误: {str(e)}")
        log(f"  堆栈跟踪:\n{traceback.format_exc()}")
        return False


def run_sanity_check():
    """Sanity Check实验"""
    log("="*80)
    log("Experiment 1: Sanity Check")
    log("="*80)
    
    # 动态导入并运行Sanity Check
    import sys
    import os
    script_path = os.path.join('experiments', '01_sanity_check.py')
    
    # 创建临时模块
    import importlib.util
    spec = importlib.util.spec_from_file_location("sanity_check", script_path)
    sanity_check_module = importlib.util.module_from_spec(spec)
    
    # 重写print函数以捕获输出
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        spec.loader.exec_module(sanity_check_module)
    finally:
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
    
    # 将输出写入日志
    log("Sanity Check输出:")
    for line in output.split('\n'):
        if line.strip():
            log(f"  {line}")
    
    log("Sanity Check完成")


def run_final_experiment():
    """最终实验：AI vs Rule"""
    log("="*80)
    log("Experiment 2: Final Experiment (AI vs Rule)")
    log("="*80)
    
    # 动态导入并运行最终实验
    import sys
    import os
    script_path = os.path.join('experiments', 'final_experiment.py')
    
    # 创建临时模块
    import importlib.util
    spec = importlib.util.spec_from_file_location("final_experiment", script_path)
    final_experiment_module = importlib.util.module_from_spec(spec)
    
    # 重写print函数以捕获输出
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        spec.loader.exec_module(final_experiment_module)
    finally:
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
    
    # 将输出写入日志
    log("最终实验输出:")
    for line in output.split('\n'):
        if line.strip():
            log(f"  {line}")
    
    log("最终实验完成")


def main():
    """主函数：运行所有实验"""
    try:
        # 记录开始
        log("开始复现所有实验...")
        log()
        
        # 实验列表
        experiments = [
            ("Sanity Check", run_sanity_check),
            ("Final Experiment (AI vs Rule)", run_final_experiment),
        ]
        
        # 运行所有实验
        success_count = 0
        for exp_name, exp_func in experiments:
            if run_experiment(exp_name, exp_func):
                success_count += 1
            log()
        
        # 总结
        log("="*80)
        log("复现总结")
        log("="*80)
        log(f"成功实验: {success_count}/{len(experiments)}")
        log(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"日志文件: {log_file}")
        log()
        
        if success_count == len(experiments):
            log("✓ 所有实验复现成功！")
            print()
            print("="*80)
            print("Reproduction completed successfully")
            print("="*80)
            return 0
        else:
            log("✗ 部分实验失败，请检查日志")
            print()
            print("="*80)
            print("Reproduction failed - check logs")
            print("="*80)
            return 1
            
    except Exception as e:
        log("="*80)
        log("复现脚本发生严重错误")
        log("="*80)
        log(f"错误: {str(e)}")
        log(f"堆栈跟踪:\n{traceback.format_exc()}")
        print()
        print("="*80)
        print("Reproduction failed - critical error")
        print("="*80)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
