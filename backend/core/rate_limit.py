"""Simple in-memory rate limiter."""

from __future__ import annotations

import time
from collections import defaultdict, deque


def _now() -> float:
    return time.time()


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.hits: defaultdict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        current = _now()
        window_start = current - self.window_seconds
        queue = self.hits[key]
        while queue and queue[0] < window_start:
            queue.popleft()
        if len(queue) >= self.max_requests:
            return False
        queue.append(current)
        return True

