import collections
import itertools

import numpy as np
import pytest

from pekoe.experimental.optimizers.scheduler import Query, Scheduler


def job_from_list(size, data_list, priority_list=None):
    if priority_list is None:
        priority_list = itertools.repeat(0.0)
    for data, priority in zip(data_list, priority_list):
        yield Query(data, size=size, priority=priority)


class Diverge(RuntimeError):
    pass


def diverging_job(*args, **kwargs):
    yield from job_from_list(*args, **kwargs)
    raise Diverge()


def test_simple():
    job0 = job_from_list(3, ["a", "b"])
    job1 = job_from_list(4, ["c"])
    job2 = job_from_list(2, ["d", "e", "f"])
    jobs = iter([job0, job1, job2])
    s = Scheduler(jobs, capacity=100, fetch_capacity=200)
    assert list(s) == [
        ["a", "c", "d"],
        ["b", "e"],
        ["f"],
    ]


def test_capacity():
    job0 = job_from_list(23, ["a", "b"])
    job1 = job_from_list(24, ["c"])
    job2 = job_from_list(22, ["d", "e", "f"])
    jobs = iter([job0, job1, job2])
    s = Scheduler(jobs, capacity=50, fetch_capacity=200)
    batches = list(s)
    cnt = collections.Counter()
    for batch in batches:
        assert 1 <= len(batch) <= 2
        cnt.update(batch)
    assert cnt == collections.Counter(["a", "b", "c", "d", "e", "f"])


def test_fetch_capacity():
    job0 = job_from_list(10, ["a", "b"])
    job1 = job_from_list(10, ["c", "d"])
    job2 = job_from_list(5, ["e"])
    jobs = iter([job0, job1, job2])
    s = Scheduler(jobs, capacity=20, fetch_capacity=21)
    assert list(s) == [
        ["a", "c"],
        ["b", "d"],
        ["e"],
    ]


def test_online():
    side_effects = []

    def effectful_job(size, data_list, eff_list):
        assert len(data_list) + 1 == len(eff_list)
        side_effects.append(eff_list[0])
        for data, eff in zip(data_list, eff_list[1:]):
            yield Query(data, size=size, priority=0.0)
            side_effects.append(eff)

    job0 = effectful_job(23, ["a", "b"], ["0", "A", "B"])
    job1 = effectful_job(24, ["c"], ["1", "C"])
    job_trivial = effectful_job(None, [], ["trivial"])
    job2 = effectful_job(22, ["d", "e", "f"], ["2", "D", "E", "F"])
    job3 = diverging_job(100, ["g"])
    jobs = iter([job0, job1, job_trivial, job2, job3])
    s = Scheduler(jobs, capacity=100, fetch_capacity=200)
    s = iter(s)
    assert side_effects == []
    assert next(s) == ["a", "c", "d"]
    assert set(side_effects) == {"0", "1", "trivial", "2"}
    del side_effects[:]
    assert next(s) == ["b", "e"]
    assert set(side_effects) == {"A", "C", "D"}
    del side_effects[:]
    assert next(s) == ["f"]
    assert set(side_effects) == {"B", "E"}
    del side_effects[:]
    assert next(s) == ["g"]
    assert set(side_effects) == {"F"}
    del side_effects[:]
    with pytest.raises(Diverge):
        next(s)


def get_large_jobs_spec():
    rs = np.random.RandomState(seed=0)
    jobs_spec = []
    total_cost = 0
    for _ in range(30):
        size = rs.randint(10, 20)
        length = rs.randint(10, 40)
        jobs_spec.append((size, length))
        total_cost += size * length
    return jobs_spec, total_cost


def test_large():
    jobs_spec, total_cost = get_large_jobs_spec()
    jobs = iter([job_from_list(size, list(range(length))) for size, length in jobs_spec])
    s = Scheduler(jobs, capacity=100, fetch_capacity=200)
    batches = list(s)
    assert 1.1 * total_cost / 100 < len(batches) < 1.3 * total_cost / 100


def test_large_priority():
    rs = np.random.RandomState(seed=1)
    jobs_spec, total_cost = get_large_jobs_spec()
    jobs = iter(
        [
            job_from_list(
                size,
                list(range(length)),
                # job with decreasing priority should improve efficiency
                (np.arange(length)[::-1] / length + 0.1 * rs.normal(size=length)).tolist(),
            )
            for size, length in jobs_spec
        ]
    )
    s = Scheduler(jobs, capacity=100, fetch_capacity=200)
    batches = list(s)
    assert total_cost / 100 <= len(batches) < 1.1 * total_cost / 100
