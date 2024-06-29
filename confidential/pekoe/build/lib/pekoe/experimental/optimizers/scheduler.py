import dataclasses
from logging import getLogger
from typing import Any, Generator, Iterable, Iterator, List, Optional

from . import knapsack

logger = getLogger(__name__)


@dataclasses.dataclass
class Query:
    data: Any
    size: int
    priority: float


Job = Iterator[Query]


@dataclasses.dataclass
class RunningJob:
    query: Query
    job: Job


_JobStarter = Generator[List[RunningJob], int, None]  # yield  # send


def _job_starter(jobs: Iterable[Job]) -> _JobStarter:
    buffer_size = yield []
    new_jobs: List[RunningJob] = []
    for job in jobs:
        try:
            query = next(job)
        except StopIteration:
            continue
        while buffer_size < query.size:
            buffer_size = yield new_jobs
            new_jobs = []
        buffer_size -= query.size
        new_jobs.append(RunningJob(query, job))
    yield new_jobs


class Scheduler:

    # TODO: Rename `fetch_capacity`.
    def __init__(
        self,
        jobs: Iterable[Job],
        capacity: int,
        fetch_capacity: int,
    ):
        co = _job_starter(jobs)
        self._running_jobs = next(co)
        self._pending_jobs: Optional[_JobStarter] = co
        self._capacity = capacity
        self._fetch_capacity = fetch_capacity

    def __iter__(self) -> Iterator[List[Any]]:
        while True:
            self._fetch()
            if not self._running_jobs:
                if self._pending_jobs is None:
                    break
                else:
                    raise RuntimeError("fetch_capacity < query.size")

            batch_indices = self._get_batch()  # should be sorted
            yield [self._running_jobs[i].query.data for i in batch_indices]
            for i in reversed(batch_indices):  # may delete index i
                x = self._running_jobs[i]
                try:
                    query = next(x.job)
                except StopIteration:
                    del self._running_jobs[i]
                else:
                    x.query = query

    def _fetch(self) -> None:
        if self._pending_jobs is None:
            return
        sizes = [x.query.size for x in self._running_jobs]
        try:
            jobs = self._pending_jobs.send(self._fetch_capacity - sum(sizes))
        except StopIteration:
            self._pending_jobs = None
        else:
            self._running_jobs.extend(jobs)

    def _get_batch(self) -> List[int]:
        """Return sorted batch indices"""
        queries = [x.query for x in self._running_jobs]
        if self._pending_jobs is not None:
            sizes = knapsack.knapsack_approx(
                self._capacity,
                {i: query.size for i, query in enumerate(queries)},
            )
            batch = list(sizes.keys())
        else:
            # prioritized
            capacity = self._capacity
            batch = []
            keys = sorted(list(range(len(queries))), key=lambda k: -queries[k].priority)
            for k in keys:
                size = queries[k].size
                if size <= capacity:
                    capacity -= size
                    batch.append(k)
            sizes = {k: queries[k].size for k in batch}  # for print
        logger.info(f"{sum(sizes.values())} / {self._capacity} ({sizes})")

        if not batch:
            raise RuntimeError("capacity < query.size")
        return sorted(batch)
