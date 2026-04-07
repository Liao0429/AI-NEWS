import sys
sys.path.append('.')

import os
import json
import uuid
import time
import subprocess
from datetime import datetime
from typing import Dict, Optional, Any


def get_git_commit_hash() -> Optional[str]:
    """获取git commit hash（如果可用）"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


class ExperimentLogger:
    """
    实验日志系统
    
    记录实验元数据，确保可复现性
    """
    
    def __init__(self, log_file: str = "results/experiment_metadata.json"):
        """
        初始化实验日志系统
        
        Args:
            log_file: 日志文件路径
        """
        self.log_file = log_file
        self.experiments = []
        self.current_experiment_id = None
        
        # 确保目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 加载现有日志
        self._load_logs()
    
    def _load_logs(self) -> None:
        """加载现有日志"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'experiments' in data:
                        self.experiments = data['experiments']
                print(f"✓ 加载了 {len(self.experiments)} 条实验记录")
            except Exception as e:
                print(f"⚠️  加载日志失败: {e}")
                self.experiments = []
    
    def _save_logs(self) -> None:
        """保存日志"""
        try:
            data = {
                "experiments": self.experiments,
                "last_updated": datetime.now().isoformat()
            }
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✓ 实验日志已保存")
        except Exception as e:
            print(f"⚠️  保存日志失败: {e}")
    
    def start_experiment(self, 
                        model_type: str,
                        model_name: Optional[str] = None,
                        temperature: Optional[float] = None,
                        confidence_threshold: Optional[float] = None,
                        dataset_name: Optional[str] = None,
                        sample_count: Optional[int] = None,
                        random_seed: Optional[int] = None,
                        experiment_type: Optional[str] = None,
                        prompt: Optional[str] = None,
                        api_version: Optional[str] = None) -> str:
        """
        开始一个新实验（增强版）
        
        Args:
            model_type: 模型类型（keyword/ml/llm）
            model_name: LLM名称（如gpt-4, kimi）
            temperature: 温度参数
            confidence_threshold: 置信度阈值
            dataset_name: 数据集名称
            sample_count: 样本数量
            random_seed: 随机种子
            experiment_type: 实验类型
            prompt: Prompt内容
            api_version: API版本
        
        Returns:
            experiment_id: 实验ID
        """
        experiment_id = str(uuid.uuid4())
        self.current_experiment_id = experiment_id
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 获取git commit hash
        git_commit_hash = get_git_commit_hash()
        
        experiment = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            
            # 模型信息
            "model_info": {
                "model_type": model_type,
                "model_name": model_name
            },
            
            # 参数
            "parameters": {
                "temperature": temperature,
                "confidence_threshold": confidence_threshold
            },
            
            # 数据
            "data_info": {
                "dataset_name": dataset_name,
                "sample_count": sample_count
            },
            
            # 实验设置
            "experiment_info": {
                "random_seed": random_seed,
                "experiment_type": experiment_type,
                "prompt": prompt,
                "api_version": api_version
            },
            
            # 代码版本
            "code_info": {
                "git_commit_hash": git_commit_hash
            },
            
            # 时间
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            
            # 状态
            "status": "running",
            
            # 结果
            "results": {}
        }
        
        self.experiments.append(experiment)
        self._save_logs()
        
        print()
        print("="*80)
        print(f"实验开始: {experiment_type or 'Unnamed Experiment'}")
        print("="*80)
        print(f"实验ID: {experiment_id}")
        print(f"运行ID: {run_id}")
        print(f"时间戳: {experiment['timestamp']}")
        print(f"模型类型: {model_type}")
        if model_name:
            print(f"模型名称: {model_name}")
        if temperature is not None:
            print(f"温度: {temperature}")
        if confidence_threshold is not None:
            print(f"置信度阈值: {confidence_threshold}")
        if dataset_name:
            print(f"数据集: {dataset_name}")
        if sample_count:
            print(f"样本数量: {sample_count}")
        if random_seed:
            print(f"随机种子: {random_seed}")
        if git_commit_hash:
            print(f"Git commit: {git_commit_hash}")
        print("="*80)
        print()
        
        return experiment_id
    
    def update_experiment(self, 
                         experiment_id: Optional[str] = None,
                         results: Optional[Dict] = None,
                         status: Optional[str] = None) -> None:
        """
        更新实验信息
        
        Args:
            experiment_id: 实验ID（如果为None，使用当前实验）
            results: 实验结果
            status: 实验状态
        """
        exp_id = experiment_id or self.current_experiment_id
        if not exp_id:
            print("⚠️  没有活动的实验")
            return
        
        # 找到实验
        for exp in self.experiments:
            if exp['experiment_id'] == exp_id:
                if results is not None:
                    exp['results'].update(results)
                if status is not None:
                    exp['status'] = status
                break
        
        self._save_logs()
    
    def end_experiment(self, 
                      experiment_id: Optional[str] = None,
                      results: Optional[Dict] = None) -> None:
        """
        结束实验
        
        Args:
            experiment_id: 实验ID（如果为None，使用当前实验）
            results: 实验结果
        """
        exp_id = experiment_id or self.current_experiment_id
        if not exp_id:
            print("⚠️  没有活动的实验")
            return
        
        # 找到实验
        for exp in self.experiments:
            if exp['experiment_id'] == exp_id:
                exp['end_time'] = datetime.now().isoformat()
                exp['status'] = "completed"
                if results is not None:
                    exp['results'].update(results)
                
                print()
                print("="*80)
                # 向后兼容旧格式
                if 'experiment_info' in exp:
                    exp_type = exp['experiment_info'].get('experiment_type', 'Unnamed Experiment')
                else:
                    exp_type = exp.get('experiment_type', 'Unnamed Experiment')
                print(f"实验完成: {exp_type}")
                print("="*80)
                print(f"实验ID: {exp_id}")
                # 向后兼容旧格式
                if 'run_id' in exp:
                    print(f"运行ID: {exp['run_id']}")
                print(f"开始时间: {exp['start_time']}")
                print(f"结束时间: {exp['end_time']}")
                print(f"状态: {exp['status']}")
                # 向后兼容旧格式
                if 'model_info' in exp:
                    print(f"模型类型: {exp['model_info']['model_type']}")
                    if exp['model_info']['model_name']:
                        print(f"模型名称: {exp['model_info']['model_name']}")
                else:
                    if 'model' in exp:
                        print(f"模型: {exp['model']}")
                # 向后兼容旧格式
                if 'data_info' in exp:
                    if exp['data_info']['dataset_name']:
                        print(f"数据集: {exp['data_info']['dataset_name']}")
                    if exp['data_info']['sample_count']:
                        print(f"样本数量: {exp['data_info']['sample_count']}")
                if exp['results']:
                    print("结果:")
                    for key, value in exp['results'].items():
                        print(f"  {key}: {value}")
                print("="*80)
                print()
                break
        
        self._save_logs()
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """
        获取实验记录
        
        Args:
            experiment_id: 实验ID
        
        Returns:
            experiment: 实验记录
        """
        for exp in self.experiments:
            if exp['experiment_id'] == experiment_id:
                return exp
        return None
    
    def list_experiments(self, limit: int = 10) -> None:
        """
        列出最近的实验
        
        Args:
            limit: 列出的数量
        """
        print()
        print("="*80)
        print(f"最近 {min(limit, len(self.experiments))} 条实验记录")
        print("="*80)
        print()
        
        for i, exp in enumerate(reversed(self.experiments[-limit:])):
            # 向后兼容旧格式
            if 'experiment_info' in exp:
                exp_type = exp['experiment_info'].get('experiment_type', 'Unnamed')
            else:
                exp_type = exp.get('experiment_type', 'Unnamed')
            
            print(f"[{i+1}] {exp_type}")
            print(f"    ID: {exp['experiment_id']}")
            
            # 向后兼容旧格式
            if 'run_id' in exp:
                print(f"    Run ID: {exp['run_id']}")
            
            # 向后兼容旧格式
            if 'model_info' in exp:
                print(f"    模型: {exp['model_info']['model_type']}")
                if exp['model_info']['model_name']:
                    print(f"    模型名称: {exp['model_info']['model_name']}")
            else:
                if 'model' in exp:
                    print(f"    模型: {exp['model']}")
            
            # 向后兼容旧格式
            if 'data_info' in exp:
                if exp['data_info'].get('dataset_name'):
                    print(f"    数据集: {exp['data_info']['dataset_name']}")
                if exp['data_info'].get('sample_count'):
                    print(f"    样本数量: {exp['data_info']['sample_count']}")
            
            print(f"    时间: {exp['start_time']}")
            print(f"    状态: {exp['status']}")
            if exp['results']:
                print(f"    结果: {exp['results']}")
            print()
        
        print("="*80)
        print()


# 全局日志器实例
_global_logger = None


def get_logger(log_file: str = "results/experiment_metadata.json") -> ExperimentLogger:
    """
    获取全局日志器实例
    
    Args:
        log_file: 日志文件路径
    
    Returns:
        logger: 实验日志器
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = ExperimentLogger(log_file)
    return _global_logger


if __name__ == '__main__':
    # 测试代码
    print("="*80)
    print("实验日志系统测试（增强版）")
    print("="*80)
    print()
    
    logger = ExperimentLogger()
    
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
    
    # 开始实验（增强版）
    experiment_id = logger.start_experiment(
        model_type="llm",
        model_name="kimi-k2",
        temperature=0.1,
        confidence_threshold=0.6,
        dataset_name="From_News_to_Forecast",
        sample_count=150,
        random_seed=42,
        experiment_type="test_experiment",
        prompt=sample_prompt,
        api_version="v1"
    )
    
    # 模拟实验运行
    print("模拟实验运行中...")
    time.sleep(2)
    
    # 更新结果
    logger.update_experiment(
        results={
            "llm_return": 0.0037,
            "keyword_return": 0.0000,
            "p_value": 0.1468
        }
    )
    
    # 结束实验
    logger.end_experiment(
        results={
            "llm_return": 0.0037,
            "keyword_return": 0.0000,
            "p_value": 0.1468
        }
    )
    
    # 列出实验
    logger.list_experiments(limit=5)
    
    print()
    print("="*80)
    print("测试完成！")
    print("="*80)
