from fastapi import APIRouter, HTTPException
from vintage_ai.api.background.scheduler import tasks

task_router = APIRouter(prefix="/tasks", tags=["Tasks"])


@task_router.get("/{task_id}")
def task_status(task_id: str):
    state = tasks.get(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, "state": state}
