"""
Rate limiting utilities for API calls with exponential backoff and usage monitoring.
"""

import time
import random
import math
from typing import Dict, Any


class RateLimiter:
    """Advanced rate limiting with exponential backoff and usage monitoring."""
    
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
        """Calculate exponential backoff delay with jitter."""
        # Exponential backoff: base_delay * (2^attempt)
        delay = self.base_delay * (2 ** attempt)
        
        # Cap at max_delay
        delay = min(delay, self.max_delay)
        
        # Add jitter (±25% randomness)
        jitter = random.uniform(0.75, 1.25)
        delay *= jitter
        
        return delay
    
    def wait_if_needed(self, min_interval: float = 0.1):
        """Wait if needed to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Determine if we should retry based on the error and attempt count."""
        if attempt >= self.max_retries:
            return False
        
        # Retry on rate limit errors
        if hasattr(error, 'status_code'):
            if error.status_code == 429:  # Rate limit
                self.usage_stats['rate_limits_hit'] += 1
                return True
            elif error.status_code >= 500:  # Server errors
                return True
        
        # Retry on connection errors
        if "connection" in str(error).lower() or "timeout" in str(error).lower():
            return True
        
        return False
    
    def get_retry_delay(self, attempt: int) -> float:
        """Get the delay before the next retry attempt."""
        if attempt >= self.max_retries:
            return 0
        
        delay = self.calculate_delay(attempt)
        self.usage_stats['retries_performed'] += 1
        return delay
    
    def update_usage_from_headers(self, response_headers: Dict[str, Any]):
        """Update usage statistics from response headers."""
        self.usage_stats['requests_made'] += 1
        
        # Estimate cost (GPT-4o pricing)
        if 'x-usage-tokens' in response_headers:
            tokens_used = int(response_headers['x-usage-tokens'])
            self.usage_stats['tokens_used'] += tokens_used
            # Calculate cost based on model pricing
            cost = (tokens_used / 1000) * self.cost_per_token
            self.usage_stats['cost_estimate'] += cost
    
    def get_usage_summary(self) -> str:
        """Get formatted usage summary."""
        return f"""
📊 API Usage Summary:
  • Requests made: {self.usage_stats['requests_made']}
  • Tokens used: {self.usage_stats['tokens_used']:,}
  • Estimated cost: ${self.usage_stats['cost_estimate']:.4f}
  • Rate limits hit: {self.usage_stats['rate_limits_hit']}
  • Retries performed: {self.usage_stats['retries_performed']}
"""
    
    def reset_stats(self):
        """Reset usage statistics."""
        self.usage_stats = {
            'requests_made': 0,
            'tokens_used': 0,
            'cost_estimate': 0.0,
            'rate_limits_hit': 0,
            'retries_performed': 0
        }

