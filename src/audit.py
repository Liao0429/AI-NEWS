import sys
sys.path.append('.')

import os
import json
from datetime import datetime
from typing import Dict, List, Optional


class ExperimentAuditor:
    """
    实验审计模块
    
    检查实验中的潜在问题：
    1. 是否存在未来信息泄露
    2. 是否所有策略结果完全相同（可能bug）
    3. 样本数量是否过小（<30）
    4. 是否缺少p-value
    5. 是否未记录随机种子
    """
    
    def __init__(self, log_file: str = "results/experiment_metadata.json", report_file: str = "results/audit_report.txt"):
        """
        初始化实验审计模块
        
        Args:
            log_file: 实验日志文件路径
            report_file: 审计报告文件路径
        """
        self.log_file = log_file
        self.report_file = report_file
        self.experiments = []
        
        # 确保目录存在
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        # 加载实验数据
        self._load_experiments()
    
    def _load_experiments(self) -> None:
        """
        加载实验数据
        """
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'experiments' in data:
                        self.experiments = data['experiments']
                print(f"✓ 加载了 {len(self.experiments)} 条实验记录")
            except Exception as e:
                print(f"⚠️  加载实验数据失败: {e}")
                self.experiments = []
    
    def audit_experiment(self, experiment: Dict) -> List[str]:
        """
        审计单个实验
        
        Args:
            experiment: 实验记录
        
        Returns:
            问题列表
        """
        issues = []
        
        # 1. 检查是否存在未来信息泄露
        # 检查prompt中是否包含未来信息相关内容
        prompt = None
        if 'experiment_info' in experiment:
            prompt = experiment['experiment_info'].get('prompt')
        elif 'prompt' in experiment:
            prompt = experiment['prompt']
        
        if prompt:
            # 检查prompt是否包含可能暗示未来信息的内容
            future_keywords = ['future', 'next', 'tomorrow', '预测', '未来', '将要']
            for keyword in future_keywords:
                if keyword.lower() in prompt.lower():
                    issues.append("可能存在未来信息泄露: Prompt中包含暗示未来的词汇")
                    break
        
        # 2. 检查是否所有策略结果完全相同（可能bug）
        results = experiment.get('results', {})
        if results:
            # 检查策略结果（支持新旧命名）
            strategy_keys = ['keyword_return', 'hash_return', 'llm_return', 'rule_return']
            strategy_values = []
            for key in strategy_keys:
                if key in results:
                    strategy_values.append(results[key])
            
            if len(strategy_values) > 1:
                first_value = strategy_values[0]
                if all(value == first_value for value in strategy_values):
                    issues.append("所有策略结果完全相同，可能存在bug")
        
        # 3. 检查样本数量是否过小（<30）
        sample_count = None
        if 'data_info' in experiment:
            sample_count = experiment['data_info'].get('sample_count')
        elif 'sample_count' in experiment:
            sample_count = experiment.get('sample_count')
        
        if sample_count and sample_count < 30:
            issues.append(f"样本数量过小: {sample_count} < 30")
        
        # 4. 检查是否缺少p-value
        if 'p_value' not in results:
            issues.append("缺少p-value")
        
        # 5. 检查是否未记录随机种子
        random_seed = None
        if 'experiment_info' in experiment:
            random_seed = experiment['experiment_info'].get('random_seed')
        elif 'random_seed' in experiment:
            random_seed = experiment.get('random_seed')
        
        if random_seed is None:
            issues.append("未记录随机种子")
        
        return issues
    
    def audit_all_experiments(self) -> Dict[str, List[str]]:
        """
        审计所有实验
        
        Returns:
            实验ID到问题列表的映射
        """
        audit_results = {}
        
        for experiment in self.experiments:
            experiment_id = experiment.get('experiment_id')
            if experiment_id:
                issues = self.audit_experiment(experiment)
                if issues:
                    audit_results[experiment_id] = issues
        
        return audit_results
    
    def generate_report(self) -> str:
        """
        生成审计报告
        
        Returns:
            审计报告内容
        """
        audit_results = self.audit_all_experiments()
        
        report = []
        report.append("=" * 80)
        report.append("实验审计报告")
        report.append("=" * 80)
        report.append(f"生成时间: {datetime.now().isoformat()}")
        report.append(f"审计实验数量: {len(self.experiments)}")
        report.append(f"发现问题的实验数量: {len(audit_results)}")
        report.append("")
        
        if audit_results:
            report.append("问题详情:")
            report.append("-" * 80)
            
            for experiment_id, issues in audit_results.items():
                report.append(f"实验ID: {experiment_id}")
                for issue in issues:
                    report.append(f"  ⚠️  {issue}")
                report.append("")
        else:
            report.append("✅ 未发现任何问题")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self) -> None:
        """
        保存审计报告到文件
        """
        report = self.generate_report()
        
        try:
            with open(self.report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✓ 审计报告已保存到: {self.report_file}")
        except Exception as e:
            print(f"⚠️  保存审计报告失败: {e}")
    
    def run_audit(self) -> None:
        """
        运行审计并输出结果
        """
        print("=" * 80)
        print("开始实验审计")
        print("=" * 80)
        
        audit_results = self.audit_all_experiments()
        
        if audit_results:
            print(f"发现问题的实验数量: {len(audit_results)}")
            print()
            print("问题详情:")
            print("-" * 80)
            
            for experiment_id, issues in audit_results.items():
                print(f"实验ID: {experiment_id}")
                for issue in issues:
                    print(f"  ⚠️  {issue}")
                print()
        else:
            print("✅ 未发现任何问题")
        
        # 保存报告
        self.save_report()
        
        print("=" * 80)
        print("审计完成")
        print("=" * 80)


def get_auditor() -> ExperimentAuditor:
    """
    获取全局审计器实例
    
    Returns:
        auditor: 实验审计器
    """
    global _global_auditor
    if _global_auditor is None:
        _global_auditor = ExperimentAuditor()
    return _global_auditor


# 全局审计器实例
_global_auditor = None


if __name__ == '__main__':
    # 测试代码
    print("=" * 80)
    print("实验审计模块测试")
    print("=" * 80)
    print()
    
    auditor = ExperimentAuditor()
    auditor.run_audit()
    
    print()
    print("=" * 80)
    print("测试完成！")
    print("=" * 80)