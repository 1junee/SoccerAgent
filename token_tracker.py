import json
import time
from datetime import datetime
import os
import ast
from dotenv import load_dotenv
load_dotenv()

class DeepSeekTokenTracker:
    def __init__(self, output_file="deepseek_usage_log.json"):
        self.usage_log = []
        self.output_file = output_file
        self.total_cost = 0.0
        
        # DeepSeek Chat 가격
        self.rates = {
            "input": 0.56,   # per 1M tokens
            "output": 1.68   # per 1M tokens
        }
    
    def log_api_call(self, response, query_info=None, step_info=None):
        """DeepSeek API response에서 토큰 사용량 추출하여 로깅"""
        try:
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            
            # 비용 계산
            cost = (input_tokens / 1000000 * self.rates["input"] + 
                   output_tokens / 1000000 * self.rates["output"])
            
            self.total_cost += cost
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'cost_usd': round(cost, 6),
                'cumulative_cost': round(self.total_cost, 6),
                'query_info': query_info,
                'step_info': step_info
            }
            
            self.usage_log.append(log_entry)
            
            # 실시간으로 파일에 저장
            self.save_log()
            
            return cost
            
        except Exception as e:
            print(f"Error logging API call: {e}")
            return 0
    
    def save_log(self):
        """로그를 파일에 저장"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_cost': self.total_cost,
                'total_calls': len(self.usage_log),
                'usage_log': self.usage_log
            }, f, indent=2, ensure_ascii=False)
    
    def get_statistics(self):
        """통계 정보 반환"""
        if not self.usage_log:
            return {"message": "No API calls logged yet"}
        
        total_input = sum(entry['input_tokens'] for entry in self.usage_log)
        total_output = sum(entry['output_tokens'] for entry in self.usage_log)
        
        return {
            'total_calls': len(self.usage_log),
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'total_cost_usd': round(self.total_cost, 6),
            'avg_input_per_call': round(total_input / len(self.usage_log), 2),
            'avg_output_per_call': round(total_output / len(self.usage_log), 2),
            'avg_cost_per_call': round(self.total_cost / len(self.usage_log), 6)
        }
    
    def print_statistics(self):
        """통계를 출력"""
        stats = self.get_statistics()
        if 'message' in stats:
            print(stats['message'])
            return
        
        print("=== DeepSeek API Usage Statistics ===")
        print(f"Total API Calls: {stats['total_calls']}")
        print(f"Total Input Tokens: {stats['total_input_tokens']:,}")
        print(f"Total Output Tokens: {stats['total_output_tokens']:,}")
        print(f"Total Cost: ${stats['total_cost_usd']:.6f}")
        print(f"Average Input per Call: {stats['avg_input_per_call']}")
        print(f"Average Output per Call: {stats['avg_output_per_call']}")
        print(f"Average Cost per Call: ${stats['avg_cost_per_call']:.6f}")
        
    def estimate_full_cost(self, sample_size, total_size):
        """10% 샘플 기반 전체 비용 예측"""
        if not self.usage_log:
            return 0
        
        scaling_factor = total_size / sample_size
        estimated_total_cost = self.total_cost * scaling_factor
        
        print(f"\n=== Cost Estimation ===")
        print(f"Sample Size: {sample_size}")
        print(f"Total Dataset Size: {total_size}")
        print(f"Sample Cost: ${self.total_cost:.6f}")
        print(f"Estimated Full Cost: ${estimated_total_cost:.2f}")
        print(f"Scaling Factor: {scaling_factor:.2f}x")
        
        return estimated_total_cost

token_tracker = DeepSeekTokenTracker()