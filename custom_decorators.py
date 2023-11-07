from functools import wraps
import time

def log_runtime_metrics(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Measure the start time
        start_time = time.time()

        # Call the original function
        result = func(*args, **kwargs)

        # Measure the end time
        end_time = time.time()

        # Calculate the runtime
        runtime = end_time - start_time

        # Log the runtime (you can customize how you log this information)
        print(f"Runtime for {func.__name__}: {runtime} seconds")

        return result

    return wrapper