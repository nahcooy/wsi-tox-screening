from __future__ import annotations

import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

JOB_TIMEOUT_SECONDS = 300

from app.schemas.preprocess import JobStatus


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalJobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()

    def start(
        self,
        *,
        command: list[str],
        cwd: Path,
        log_path: Path,
        env: dict[str, str],
        metadata: dict,
    ) -> JobStatus:
        job_id = uuid.uuid4().hex
        log_path.parent.mkdir(parents=True, exist_ok=True)

        job = JobStatus(
            job_id=job_id,
            status="PENDING",
            command=command,
            cwd=str(cwd),
            log_path=str(log_path),
            metadata=metadata,
        )
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run,
            args=(job_id, command, cwd, log_path, env),
            daemon=True,
        )
        thread.start()
        return job

    def _run(
        self,
        job_id: str,
        command: list[str],
        cwd: Path,
        log_path: Path,
        env: dict[str, str],
    ) -> None:
        self._update(job_id, status="RUNNING", started_at=utc_now())
        try:
            with log_path.open("a", encoding="utf-8", buffering=1) as log:
                log.write(f"[JOB {job_id}] started_at={utc_now()}\n")
                log.write("[COMMAND] " + " ".join(command) + "\n\n")
                process = subprocess.Popen(
                    command,
                    cwd=str(cwd),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                with self._lock:
                    self._processes[job_id] = process

                def _kill_on_timeout() -> None:
                    log.write(f"\n[TIMEOUT] Job killed after {JOB_TIMEOUT_SECONDS}s\n")
                    process.kill()

                timer = threading.Timer(JOB_TIMEOUT_SECONDS, _kill_on_timeout)
                timer.start()
                try:
                    assert process.stdout is not None
                    for line in process.stdout:
                        log.write(line)
                    returncode = process.wait()
                finally:
                    timer.cancel()

                if returncode == -9:
                    self._update(
                        job_id,
                        status="FAILED",
                        returncode=returncode,
                        finished_at=utc_now(),
                        error=f"Job timed out after {JOB_TIMEOUT_SECONDS}s",
                    )
                elif returncode == 0:
                    self._update(
                        job_id,
                        status="COMPLETED",
                        returncode=returncode,
                        finished_at=utc_now(),
                    )
                else:
                    self._update(
                        job_id,
                        status="FAILED",
                        returncode=returncode,
                        finished_at=utc_now(),
                        error=f"Process exited with code {returncode}",
                    )
        except Exception as exc:
            self._update(
                job_id,
                status="FAILED",
                finished_at=utc_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            with self._lock:
                self._processes.pop(job_id, None)

    def _update(self, job_id: str, **updates: object) -> None:
        with self._lock:
            job = self._jobs[job_id]
            data = job.model_dump()
            data.update(updates)
            self._jobs[job_id] = JobStatus(**data)

    def get(self, job_id: str) -> JobStatus | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> JobStatus | None:
        with self._lock:
            process = self._processes.get(job_id)
        if process is not None and process.poll() is None:
            process.terminate()
            self._update(job_id, status="CANCELLED", finished_at=utc_now())
        return self.get(job_id)


job_manager = LocalJobManager()

