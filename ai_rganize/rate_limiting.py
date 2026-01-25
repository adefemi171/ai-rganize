"""Rate limiting with exponential backoff."""

import time
import random
from typing import Dict, Any


class RateLimiter:
    def __init__(self, max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 300.0, cost_per_token: float = 0.01):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.cost_per_token = cost_per_token
        self.attempt = 0
        self.last_request_time = 0
        self.usage_stats = {
            'requests_made': 0,
            'tokens_used': 0,
            'cost_estimate': 0.0,
            'rate_limits_hit': 0,
            'retries_performed': 0
        }
    
    def calculate_delay(self, attempt: int) -> float:
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        return delay * random.uniform(0.75, 1.25)  # Add jitter
    
    def wait_if_needed(self, min_interval: float = 0.1):
        elapsed = time.time() - self.last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()
    
    def should_retry(self, attempt: int, error: Exception) -> bool:
        if attempt >= self.max_retries:
            return False
        
        if hasattr(error, 'status_code'):
            if error.status_code == 429:
                self.usage_stats['rate_limits_hit'] += 1
                return True
            if error.status_code >= 500:
                return True
        
        error_str = str(error).lower()
        return "connection" in error_str or "timeout" in error_str
    
    def get_retry_delay(self, attempt: int) -> float:
        if attempt >= self.max_retries:
            return 0
        self.usage_stats['retries_performed'] += 1
        return self.calculate_delay(attempt)
    
    def update_usage_from_headers(self, response_headers: Dict[str, Any]):
        self.usage_stats['requests_made'] += 1
        if 'x-usage-tokens' in response_headers:
            tokens_used = int(response_headers['x-usage-tokens'])
            self.usage_stats['tokens_used'] += tokens_used
            self.usage_stats['cost_estimate'] += (tokens_used / 1000) * self.cost_per_token
    
    def get_usage_summary(self) -> str:
        return f"""
📊 API Usage Summary:
  • Requests made: {self.usage_stats['requests_made']}
  • Tokens used: {self.usage_stats['tokens_used']:,}
  • Estimated cost: ${self.usage_stats['cost_estimate']:.4f}
  • Rate limits hit: {self.usage_stats['rate_limits_hit']}
  • Retries performed: {self.usage_stats['retries_performed']}
"""
    
    def reset_stats(self):
        self.usage_stats = {
            'requests_made': 0,
            'tokens_used': 0,
            'cost_estimate': 0.0,
            'rate_limits_hit': 0,
            'retries_performed': 0
        }

