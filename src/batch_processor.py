"""Batch Processing for MEOKCLAW — Async Job Queue

Process large batches of AI requests asynchronously with progress tracking.

Features:
- Submit batch jobs via API
- Process with configurable concurrency
- Progress tracking and status polling
- Cost aggregation per batch
- Result export (JSON, CSV)
- Retry failed items automatically
- Cancel in-flight batches

Usage:
    from batch_processor import batch_processor
    
    job_id = await batch_processor.submit(
        requests=[{"prompt": "..."}, ...],
        model="deepseek-v4-flash",
        max_concurrency=5,
    )
    
    # Poll status
    status = batch_processor.status(job_id)
    # { "completed": 45, "total": 100, "failed": 2, "cost_usd": 0.012 }
    
    # Get results
    results = batch_processor.results(job_id)
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchItem:
    id: str
    prompt: str
    system_prompt: Optional[str] = None
    status: str = "pending"
    result: Optional[str] = None
    error: Optional[str] = None
    cost_usd: float = 0.0
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0


@dataclass
class BatchJob:
    id: str
    status: JobStatus
    model: str
    items: List[BatchItem]
    max_concurrency: int
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    total_cost_usd: float = 0.0
    total_latency_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    callback_url: Optional[str] = None


class BatchProcessor:
    """Async batch processing with progress tracking."""

    def __init__(self, max_jobs: int = 100):
        self._jobs: Dict[str, BatchJob] = {}
        self._max_jobs = max_jobs
        self._running_tasks: Dict[str, asyncio.Task] = {}

    async def submit(
        self,
        requests: List[Dict[str, str]],
        model: str = "deepseek-v4-flash",
        max_concurrency: int = 5,
        system_prompt: Optional[str] = None,
        callback_url: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Submit a new batch job."""
        job_id = f"batch_{uuid.uuid4().hex[:12]}"

        items = [
            BatchItem(
                id=f"{job_id}_item_{i}",
                prompt=req.get("prompt", req.get("message", "")),
                system_prompt=system_prompt or req.get("system_prompt"),
            )
            for i, req in enumerate(requests)
        ]

        job = BatchJob(
            id=job_id,
            status=JobStatus.PENDING,
            model=model,
            items=items,
            max_concurrency=max_concurrency,
            created_at=time.time(),
            callback_url=callback_url,
            metadata=metadata or {},
        )

        self._jobs[job_id] = job

        # Prune old jobs
        if len(self._jobs) > self._max_jobs:
            oldest = sorted(self._jobs.keys(), key=lambda k: self._jobs[k].created_at)[0]
            del self._jobs[oldest]

        # Start processing
        asyncio.create_task(self._process_job(job_id))

        return job_id

    async def _process_job(self, job_id: str):
        """Process a batch job."""
        job = self._jobs.get(job_id)
        if not job:
            return

        job.status = JobStatus.RUNNING
        job.started_at = time.time()

        semaphore = asyncio.Semaphore(job.max_concurrency)

        async def process_item(item: BatchItem):
            async with semaphore:
                if job.status == JobStatus.CANCELLED:
                    return

                item.start_time = time.time()
                item.status = "running"

                try:
                    # Call inference
                    from dual_brain_orchestrator import DualBrainOrchestrator
                    orch = DualBrainOrchestrator()

                    messages = []
                    if item.system_prompt:
                        messages.append({"role": "system", "content": item.system_prompt})
                    messages.append({"role": "user", "content": item.prompt})

                    result = await orch.think(item.prompt, None)

                    item.result = result.get("text", str(result)) if isinstance(result, dict) else str(result)
                    item.cost_usd = result.get("cost_usd", 0.0) if isinstance(result, dict) else 0.0
                    item.tokens_in = result.get("tokens_in", 0) if isinstance(result, dict) else 0
                    item.tokens_out = result.get("tokens_out", 0) if isinstance(result, dict) else 0
                    item.status = "completed"

                except Exception as e:
                    item.error = str(e)
                    item.status = "failed"
                    item.retry_count += 1

                    # Retry once
                    if item.retry_count < 2:
                        await asyncio.sleep(1)
                        return await process_item(item)

                finally:
                    item.end_time = time.time()
                    item.latency_ms = int((item.end_time - item.start_time) * 1000)

        # Process all items
        tasks = [asyncio.create_task(process_item(item)) for item in job.items]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate totals
        job.total_cost_usd = sum(item.cost_usd for item in job.items)
        job.total_latency_ms = sum(item.latency_ms for item in job.items)
        job.completed_at = time.time()

        failed = sum(1 for item in job.items if item.status == "failed")
        if failed == len(job.items):
            job.status = JobStatus.FAILED
        else:
            job.status = JobStatus.COMPLETED

        # Send callback if configured
        if job.callback_url:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        job.callback_url,
                        json=self._job_to_dict(job),
                    )
            except Exception:
                pass

    def status(self, job_id: str) -> Dict[str, Any]:
        """Get job status."""
        job = self._jobs.get(job_id)
        if not job:
            return {"error": "Job not found"}

        completed = sum(1 for item in job.items if item.status == "completed")
        failed = sum(1 for item in job.items if item.status == "failed")
        running = sum(1 for item in job.items if item.status == "running")
        pending = len(job.items) - completed - failed - running

        return {
            "job_id": job_id,
            "status": job.status.value,
            "model": job.model,
            "progress": {
                "total": len(job.items),
                "completed": completed,
                "failed": failed,
                "running": running,
                "pending": pending,
                "percent": round((completed + failed) / len(job.items) * 100, 1) if job.items else 0,
            },
            "cost": {
                "total_usd": round(job.total_cost_usd, 6),
                "avg_per_item": round(job.total_cost_usd / max(len(job.items), 1), 6),
            },
            "timing": {
                "created": job.created_at,
                "started": job.started_at,
                "completed": job.completed_at,
                "duration_seconds": round((job.completed_at or time.time()) - (job.started_at or job.created_at), 1),
            },
        }

    def results(self, job_id: str, format: str = "json") -> Any:
        """Get job results."""
        job = self._jobs.get(job_id)
        if not job:
            return {"error": "Job not found"}

        items = [
            {
                "id": item.id,
                "prompt": item.prompt,
                "status": item.status,
                "result": item.result,
                "error": item.error,
                "cost_usd": item.cost_usd,
                "latency_ms": item.latency_ms,
                "tokens_in": item.tokens_in,
                "tokens_out": item.tokens_out,
            }
            for item in job.items
        ]

        if format == "csv":
            import csv
            import io
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=items[0].keys() if items else [])
            writer.writeheader()
            writer.writerows(items)
            return output.getvalue()

        return {
            "job_id": job_id,
            "items": items,
            "summary": self.status(job_id),
        }

    def cancel(self, job_id: str) -> bool:
        """Cancel a running job."""
        job = self._jobs.get(job_id)
        if not job or job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
            return False

        job.status = JobStatus.CANCELLED
        return True

    def list_jobs(self, limit: int = 50) -> List[Dict]:
        """List recent jobs."""
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [self.status(j.id) for j in jobs[:limit]]

    def _job_to_dict(self, job: BatchJob) -> Dict:
        return {
            "job_id": job.id,
            "status": job.status.value,
            "results": self.results(job.id),
        }


# Singleton
batch_processor = BatchProcessor()


if __name__ == "__main__":
    import asyncio

    async def demo():
        bp = BatchProcessor()

        requests = [
            {"prompt": "What is 2+2?"},
            {"prompt": "Explain quantum computing"},
            {"prompt": "Write a Python function to sort a list"},
        ]

        job_id = await bp.submit(requests, model="deepseek-v4-flash", max_concurrency=2)
        print(f"Submitted batch: {job_id}")

        # Poll for completion
        for _ in range(30):
            status = bp.status(job_id)
            print(f"Progress: {status['progress']['percent']}%")
            if status['status'] in ['completed', 'failed']:
                break
            await asyncio.sleep(1)

        print("\nFinal status:")
        print(json.dumps(bp.status(job_id), indent=2))

    asyncio.run(demo())
