from dataclasses import dataclass, field
from collections import defaultdict
import time
from typing import Dict, List
from threading import Lock


@dataclass
class RateLimiter:
    """Rate limiter implementation for chat interface"""
    requests_per_minute: int = 2
    window_size: int = 60  # seconds
    
    # Track request timestamps per user IP
    _requests: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    _lock: Lock = field(default_factory=Lock)

    def _clean_old_requests(self, user_ip: str) -> None:
        """Remove requests older than the window size"""
        current_time = time.time()
        with self._lock:
            self._requests[user_ip] = [
                timestamp for timestamp in self._requests[user_ip]
                if current_time - timestamp < self.window_size
            ]

    def check_rate_limit(self, user_ip: str) -> bool:
        """
        Check if user has exceeded rate limit
        Returns True if rate limit exceeded, False otherwise
        """
        self._clean_old_requests(user_ip)
        
        with self._lock:
            if len(self._requests[user_ip]) >= self.requests_per_minute:
                return True
                
            self._requests[user_ip].append(time.time())
            return False

    def get_remaining_time(self, user_ip: str) -> float:
        """Get remaining time until next request is allowed"""
        self._clean_old_requests(user_ip)
        
        with self._lock:
            if not self._requests[user_ip]:
                return 0
                
            oldest_request = min(self._requests[user_ip])
            current_time = time.time()
            time_until_reset = self.window_size - (current_time - oldest_request)
            
            return max(0, time_until_reset)