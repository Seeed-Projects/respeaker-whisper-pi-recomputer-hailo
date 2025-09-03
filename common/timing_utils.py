"""Timing utilities for performance analysis."""

import time

# Global flag to control timing display
SHOW_TIMING = False

def set_timing_display(show_timing):
    """Set whether to show timing information."""
    global SHOW_TIMING
    SHOW_TIMING = show_timing

def timed_print(message):
    """Print message only if timing display is enabled."""
    if SHOW_TIMING:
        print(message)

def time_function(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        if SHOW_TIMING:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            timed_print(f"[{func.__name__}] {end_time - start_time:.3f}s")
            return result
        else:
            return func(*args, **kwargs)
    return wrapper