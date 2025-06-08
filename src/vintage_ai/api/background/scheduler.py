import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Literal

# -----------------------------------------------------------------------------
# Thread pool (4 concurrent scrapers; tune to your CPU / I/O needs)
POOL = ThreadPoolExecutor(max_workers=4)

# -----------------------------------------------------------------------------
# Very small task registry – lives in RAM for hackathon speed
TaskState = Literal["PENDING", "SUCCESS", "FAIL"]
tasks: Dict[str, TaskState] = {}

# -----------------------------------------------------------------------------


def submit(fn: Callable, *args, **kwargs) -> str:
    """
    Wrap any blocking function (your scraper) in a thread-pool job.
    Returns a task_id so the API can let the UI poll for status.
    """
    job_id = uuid.uuid4().hex
    tasks[job_id] = "PENDING"

    def _run():
        try:
            fn(*args, **kwargs)
            tasks[job_id] = "SUCCESS"
        except Exception as exc:
            tasks[job_id] = "FAIL"
            print(f"[TASK {job_id[:8]}] crashed:", exc)

    POOL.submit(_run)
    return job_id
