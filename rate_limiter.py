"""
FIFO Rate Limiter for MCP servers.

Provides two-level rate limiting (global + per-tool) with a queue-based
approach. When rate limited, requests wait in a FIFO queue until they
can proceed or timeout.

Usage:
    from rate_limiter import RateLimiter, rate_limited

    # Create limiter (reads from rate_limits.json)
    limiter = RateLimiter()

    # Apply to tools
    @mcp.tool(...)
    @rate_limited(limiter)
    @notify_on_error
    def my_tool(...):
        ...
"""

import json
import threading
import time
from collections import deque
from functools import wraps
from pathlib import Path


class RateLimitTimeout(Exception):
    """Raised when a request times out waiting for rate limit."""

    pass


class RateLimiter:
    """
    FIFO rate limiter with two-level limiting (global + per-tool).

    Requests are processed in order. When rate limited, requests wait
    until they can proceed or timeout after max_wait_seconds.
    """

    def __init__(self, config_path: str = "rate_limits.json"):
        """
        Initialize the rate limiter.

        Args:
            config_path: Path to rate_limits.json config file.
        """
        self.config = self._load_config(config_path)
        self._global_timestamps: deque[float] = deque()
        self._tool_timestamps: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        # FIFO queue: list of (event, tool_name) tuples
        self._queue: deque[tuple[threading.Event, str]] = deque()
        self._queue_lock = threading.Lock()

    def _load_config(self, config_path: str) -> dict:
        """Load config from JSON file or return defaults."""
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {
            "max_requests_per_second": 10,
            "max_wait_seconds": 120,
            "tools": {},
        }

    def _get_tool_limit(self, tool_name: str) -> float:
        """Get rate limit for a specific tool."""
        tools = self.config.get("tools", {})
        if tool_name in tools:
            return tools[tool_name].get(
                "max_requests_per_second", self.config["max_requests_per_second"]
            )
        return self.config["max_requests_per_second"]

    def _clean_old_timestamps(self, timestamps: deque, window: float = 1.0) -> None:
        """Remove timestamps older than the window."""
        cutoff = time.time() - window
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()

    def _calculate_wait_time(self, tool_name: str) -> float:
        """
        Calculate how long until a request can proceed.

        Returns 0 if the request can proceed immediately.
        """
        global_limit = self.config["max_requests_per_second"]
        tool_limit = self._get_tool_limit(tool_name)
        now = time.time()
        wait_times = []

        # Check global limit
        self._clean_old_timestamps(self._global_timestamps)
        if len(self._global_timestamps) >= global_limit:
            oldest = self._global_timestamps[0]
            wait_times.append(oldest + 1.0 - now)

        # Check tool-specific limit
        if tool_name not in self._tool_timestamps:
            self._tool_timestamps[tool_name] = deque()
        self._clean_old_timestamps(self._tool_timestamps[tool_name])
        if len(self._tool_timestamps[tool_name]) >= tool_limit:
            oldest = self._tool_timestamps[tool_name][0]
            wait_times.append(oldest + 1.0 - now)

        return max(wait_times) if wait_times else 0.0

    def _record_call(self, tool_name: str) -> None:
        """Record a call timestamp for rate limiting."""
        now = time.time()
        self._global_timestamps.append(now)
        if tool_name not in self._tool_timestamps:
            self._tool_timestamps[tool_name] = deque()
        self._tool_timestamps[tool_name].append(now)

    def _notify_next(self) -> None:
        """Notify the next request in queue that it can try to proceed."""
        with self._queue_lock:
            if self._queue:
                self._queue[0][0].set()

    def acquire(self, tool_name: str) -> None:
        """
        Acquire rate limit permission. Blocks until allowed or raises timeout.

        Args:
            tool_name: Name of the tool being called.

        Raises:
            RateLimitTimeout: If request times out waiting for rate limit.
        """
        max_wait = self.config.get("max_wait_seconds", 120)
        start_time = time.time()

        # Create event for this request and join queue
        my_event = threading.Event()
        with self._queue_lock:
            self._queue.append((my_event, tool_name))
            # If we're first in queue, we can start trying immediately
            if len(self._queue) == 1:
                my_event.set()

        try:
            while True:
                # Wait for our turn (notified when at front of queue)
                elapsed = time.time() - start_time
                remaining = max_wait - elapsed
                if remaining <= 0:
                    raise RateLimitTimeout(
                        f"Rate limit timeout after {max_wait}s waiting for '{tool_name}'"
                    )

                # Wait for notification or periodic check
                my_event.wait(timeout=min(0.1, remaining))

                # Check if we're at front of queue
                with self._queue_lock:
                    if not self._queue or self._queue[0][0] is not my_event:
                        my_event.clear()
                        continue

                # We're at front - try to acquire
                with self._lock:
                    wait_time = self._calculate_wait_time(tool_name)
                    if wait_time <= 0:
                        # Can proceed
                        self._record_call(tool_name)
                        with self._queue_lock:
                            self._queue.popleft()
                        self._notify_next()
                        return

                # Need to wait for rate limit
                elapsed = time.time() - start_time
                remaining = max_wait - elapsed
                if remaining <= 0:
                    raise RateLimitTimeout(
                        f"Rate limit timeout after {max_wait}s waiting for '{tool_name}'"
                    )

                # Sleep until rate limit allows (or timeout)
                sleep_time = min(wait_time + 0.01, remaining)
                time.sleep(sleep_time)

        except Exception:
            # Clean up: remove from queue on any exception
            with self._queue_lock:
                self._queue = deque(
                    (e, t) for e, t in self._queue if e is not my_event
                )
            self._notify_next()
            raise

    def get_status(self) -> dict:
        """Get current rate limiter status for debugging."""
        with self._lock:
            self._clean_old_timestamps(self._global_timestamps)
            return {
                "global_calls_last_second": len(self._global_timestamps),
                "global_limit": self.config["max_requests_per_second"],
                "queue_depth": len(self._queue),
                "tool_calls": {
                    tool: len(ts)
                    for tool, ts in self._tool_timestamps.items()
                    if ts
                },
            }


def rate_limited(limiter: RateLimiter):
    """
    Decorator to apply rate limiting to a function.

    Args:
        limiter: RateLimiter instance to use.

    Usage:
        limiter = RateLimiter()

        @rate_limited(limiter)
        def my_function():
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter.acquire(func.__name__)
            return func(*args, **kwargs)

        return wrapper

    return decorator
