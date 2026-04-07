import sys
sys.path.append('.')

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Tuple, Any


class LLMOutputValidator:
    """
    LLM输出验证器
    
    检查LLM输出是否符合要求，不符合则丢弃并记录错误
    """
    
    def __init__(self, error_log_file: str = "results/error_log.csv"):
        """
        初始化验证器
        
        Args:
            error_log_file: 错误日志文件路径
        """
        self.error_log_file = error_log_file
        self.errors = []
        
        # 确保目录存在
        os.makedirs(os.path.dirname(error_log_file), exist_ok=True)
        
        # 加载现有错误日志
        self._load_error_log()
    
    def _load_error_log(self) -> None:
        """加载现有错误日志"""
        if os.path.exists(self.error_log_file):
            try:
                df = pd.read_csv(self.error_log_file)
                self.errors = df.to_dict('records')
                print(f"✓ 加载了 {len(self.errors)} 条错误记录")
            except Exception as e:
                print(f"⚠️  加载错误日志失败: {e}")
                self.errors = []
    
    def _save_error_log(self) -> None:
        """保存错误日志"""
        try:
            df = pd.DataFrame(self.errors)
            df.to_csv(self.error_log_file, index=False)
        except Exception as e:
            print(f"⚠️  保存错误日志失败: {e}")
    
    def _log_error(self, 
                   error_type: str,
                   news_text: str,
                   llm_output: Optional[str] = None,
                   error_message: str = "",
                   timestamp: Optional[str] = None) -> None:
        """
        记录错误
        
        Args:
            error_type: 错误类型
            news_text: 新闻文本
            llm_output: LLM原始输出
            error_message: 错误消息
            timestamp: 时间戳
        """
        error = {
            "timestamp": timestamp or datetime.now().isoformat(),
            "error_type": error_type,
            "news_text": news_text[:200] + "..." if len(news_text) > 200 else news_text,
            "llm_output": llm_output[:200] + "..." if llm_output and len(llm_output) > 200 else llm_output,
            "error_message": error_message
        }
        
        self.errors.append(error)
        self._save_error_log()
        
        print(f"⚠️  错误: {error_type} - {error_message}")
    
    def validate_llm_output(self, 
                            news_text: str,
                            llm_output: str,
                            check_future_info: bool = False) -> Tuple[bool, Optional[Dict]]:
        """
        验证LLM输出
        
        Args:
            news_text: 新闻文本
            llm_output: LLM原始输出
            check_future_info: 是否检查未来信息（可选）
        
        Returns:
            (is_valid, parsed_output): 是否有效，解析后的输出（如果有效）
        """
        # 检查1: 输出是否为JSON格式
        try:
            parsed_output = json.loads(llm_output)
        except json.JSONDecodeError:
            # 尝试提取JSON部分
            try:
                start_idx = llm_output.find('{')
                end_idx = llm_output.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = llm_output[start_idx:end_idx+1]
                    parsed_output = json.loads(json_str)
                else:
                    raise ValueError("无法提取JSON")
            except Exception as e:
                self._log_error(
                    error_type="INVALID_JSON_FORMAT",
                    news_text=news_text,
                    llm_output=llm_output,
                    error_message=f"JSON解析失败: {str(e)}"
                )
                return False, None
        
        # 检查2: prediction是否在{UP, DOWN}
        if "prediction" not in parsed_output:
            self._log_error(
                error_type="MISSING_PREDICTION",
                news_text=news_text,
                llm_output=llm_output,
                error_message="缺少prediction字段"
            )
            return False, None
        
        prediction = parsed_output["prediction"].upper().strip()
        if prediction not in ["UP", "DOWN"]:
            self._log_error(
                error_type="INVALID_PREDICTION",
                news_text=news_text,
                llm_output=llm_output,
                error_message=f"prediction必须是UP或DOWN，当前值: {prediction}"
            )
            return False, None
        
        # 检查3: confidence是否在[0, 1]
        if "confidence" not in parsed_output:
            self._log_error(
                error_type="MISSING_CONFIDENCE",
                news_text=news_text,
                llm_output=llm_output,
                error_message="缺少confidence字段"
            )
            return False, None
        
        try:
            confidence = float(parsed_output["confidence"])
            if confidence < 0 or confidence > 1:
                self._log_error(
                    error_type="INVALID_CONFIDENCE",
                    news_text=news_text,
                    llm_output=llm_output,
                    error_message=f"confidence必须在[0, 1]范围内，当前值: {confidence}"
                )
                return False, None
        except (ValueError, TypeError):
            self._log_error(
                error_type="INVALID_CONFIDENCE_TYPE",
                news_text=news_text,
                llm_output=llm_output,
                error_message=f"confidence必须是数字，当前值: {parsed_output['confidence']}"
            )
            return False, None
        
        # 检查4: 是否使用未来信息（可选）
        if check_future_info:
            # 这里可以添加检查未来信息的逻辑
            # 例如：检查LLM输出中是否包含未来时间点等
            # 由于这是新闻文本，通常不会包含未来信息
            # 所以这里只是一个占位，可根据需要扩展
            pass
        
        # 所有检查通过
        parsed_output["prediction"] = prediction
        parsed_output["confidence"] = confidence
        
        return True, parsed_output
    
    def get_error_summary(self) -> pd.DataFrame:
        """
        获取错误摘要
        
        Returns:
            error_summary: 错误摘要DataFrame
        """
        if not self.errors:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.errors)
        summary = df.groupby("error_type").size().reset_index(name="count")
        summary = summary.sort_values("count", ascending=False)
        
        return summary
    
    def print_error_summary(self) -> None:
        """打印错误摘要"""
        summary = self.get_error_summary()
        
        print()
        print("="*80)
        print("LLM输出错误摘要")
        print("="*80)
        print()
        
        if summary.empty:
            print("✓ 没有错误记录")
        else:
            print(summary.to_string(index=False))
            print()
            print(f"总错误数: {len(self.errors)}")
        
        print()
        print("="*80)
        print()


# 全局验证器实例
_global_validator = None


def get_validator(error_log_file: str = "results/error_log.csv") -> LLMOutputValidator:
    """
    获取全局验证器实例
    
    Args:
        error_log_file: 错误日志文件路径
    
    Returns:
        validator: LLM输出验证器
    """
    global _global_validator
    if _global_validator is None:
        _global_validator = LLMOutputValidator(error_log_file)
    return _global_validator


if __name__ == '__main__':
    # 测试代码
    print("="*80)
    print("LLM输出验证器测试")
    print("="*80)
    print()
    
    validator = LLMOutputValidator()
    
    # 测试用例
    test_cases = [
        {
            "name": "有效输出",
            "news_text": "Apple reports record earnings",
            "llm_output": '{"prediction": "UP", "confidence": 0.85, "reason": "Good news"}',
            "should_pass": True
        },
        {
            "name": "无效JSON",
            "news_text": "Apple reports record earnings",
            "llm_output": "This is not JSON",
            "should_pass": False
        },
        {
            "name": "无效prediction",
            "news_text": "Apple reports record earnings",
            "llm_output": '{"prediction": "SIDE", "confidence": 0.85}',
            "should_pass": False
        },
        {
            "name": "无效confidence",
            "news_text": "Apple reports record earnings",
            "llm_output": '{"prediction": "UP", "confidence": 1.5}',
            "should_pass": False
        }
    ]
    
    print("运行测试用例...")
    print()
    
    for i, test_case in enumerate(test_cases):
        print(f"测试用例 {i+1}: {test_case['name']}")
        print(f"  新闻: {test_case['news_text']}")
        print(f"  LLM输出: {test_case['llm_output']}")
        
        is_valid, parsed_output = validator.validate_llm_output(
            news_text=test_case['news_text'],
            llm_output=test_case['llm_output']
        )
        
        expected = "PASS" if test_case['should_pass'] else "FAIL"
        actual = "PASS" if is_valid else "FAIL"
        
        print(f"  预期: {expected}")
        print(f"  实际: {actual}")
        
        if expected == actual:
            print(f"  ✓ 测试通过")
        else:
            print(f"  ✗ 测试失败")
        
        print()
    
    # 打印错误摘要
    validator.print_error_summary()
    
    print()
    print("="*80)
    print("测试完成！")
    print("="*80)
