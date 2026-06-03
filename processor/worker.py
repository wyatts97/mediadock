"""
worker.py — Celery worker that processes media jobs from Redis queue.
Each job is dispatched to the appropriate operation pipeline.
"""

import os
import json
import logging
from celery import Celery
from operations.pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

app = Celery("media_worker", broker=REDIS_URL, backend=REDIS_URL)
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=86400,           # 24 h
    worker_prefetch_multiplier=1,   # one job at a time per worker (GPU)
    task_track_started=True,
    task_send_sent_event=True,
)


@app.task(bind=True, name="process_media", track_started=True)
def process_media(self, job: dict) -> dict:
    """
    job = {
        "job_id":    str,
        "input":     str,   # abs path inside container
        "output":    str,   # abs path inside container
        "media_type": "image" | "video",
        "operations": [...],        # from AI analysis or manual
        "settings":  {...},
    }
    """
    log.info("Starting job %s  file=%s", job["job_id"], job["input"])

    def progress(pct: int, msg: str):
        self.update_state(
            state="PROGRESS",
            meta={"progress": pct, "status": msg, "job_id": job["job_id"]},
        )

    result = run_pipeline(job, progress_cb=progress)
    log.info("Finished job %s -> %s", job["job_id"], result.get("output"))
    return result


if __name__ == "__main__":
    app.worker_main(
        argv=["worker", "--loglevel=info", "--concurrency=1", "-Q", "media"]
    )
