import dataclasses
import functools
from logging import getLogger
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import cupy
import cupy.cuda
import cupyx
import torch
from typing_extensions import Protocol  # `typing.Protocol` is from Python 3.8

from . import cupytorch
from .bfgs import BFGS
from .data import CuPyMol  # mypy
from .model import MyTeaNet
from .scheduler import Query, Scheduler

if TYPE_CHECKING:
    from pekoe.nn.models.teanet_estimator import TeaNetEstimator


logger = getLogger(__name__)


BaseOutput = Dict[str, Any]
UserInput = TypeVar("UserInput", contravariant=True)
UserOutput = TypeVar("UserOutput", covariant=True)


class UserFunc(Protocol[UserInput, UserOutput]):
    def __call__(
        self,
        x: UserInput,
        # TODO: Python 3.8
        # /,
        *,
        optimize: Callable[[CuPyMol], Generator[Query, None, BaseOutput]],
    ) -> Generator[Query, None, UserOutput]:
        ...


@dataclasses.dataclass
class _OptConfig:
    lr: float = 0.1
    maxstep: float = 0.2
    steps: int = 1000
    fmax: float = 1e-2


class ScaledBFGSCallback(Protocol):
    def __call__(
        self,
        *,
        x: cupy.ndarray,
        y: float,
        gx: cupy.ndarray,
    ) -> None:
        ...


class ScaledBFGS:
    """Scaled BFGS optimizer

    For each iteration, using the second order gradient of the objective
    function, the update step suggested by BFGS algorithm is scaled in the
    sense of scaled conjugate gradient algorithm.

    Args:
        mol: Target atoms.
        lr (float): Learning rate. See `BFGS.lr`.
        maxstep (float): The maximum distance an atom can move per
            iteration. See `ase.optimize.Optimizer`.

    """

    def __init__(
        self,
        mol: CuPyMol,
        *,
        lr: float,
        maxstep: float,
        observe_fn: Optional[ScaledBFGSCallback] = None,
    ):
        self.mol = mol
        self.bfgs = BFGS(lr=lr)
        self.maxstep = maxstep
        self.observe_fn = observe_fn

    def update(self) -> Generator[Any, Any, None]:
        mol = self.mol
        y, gx = yield mol
        x = mol.positions

        observe_fn = self.observe_fn
        if observe_fn is not None:
            observe_fn(x=x, y=y, gx=gx)

        dx = self.bfgs.step(x, gx)

        inv_stepsize = cupyx.rsqrt(cupy.square(dx).sum(1).max())
        # make stepsize <= self.maxstep
        dx *= cupy.fmin(1, self.maxstep * inv_stepsize)

        hxdx = yield dx
        gd = gx.ravel() @ dx.ravel()
        dhd = dx.ravel() @ hxdx.ravel()
        alpha = -gd / cupy.fmax(dhd, abs(gd))
        x += alpha * dx
        # The redundant `yield` works around a bug in PyTorch.
        yield

    @staticmethod
    def batch_update(
        selves: Iterable["ScaledBFGS"],
        fn: Callable[[List[CuPyMol]], Tuple[torch.Tensor, List[torch.Tensor]]],
    ) -> None:
        """Run `update` in batch

        Args:
            selves (iterable of `ScaledBFGS`): Optimizers to run.
            fn (callable): Function that takes mols and returns the tuple:
                - `y` (`Tensor`): their (stacked) potential energy; and
                - `xs` (list of `Tensor`): their positions.

        """
        upds = [self.update() for self in selves]
        del selves

        mols = [next(upd) for upd in upds]
        y, xs = fn(mols)
        y_cpu = y.detach().tolist()
        gxs = torch.autograd.grad([y], xs, [torch.ones_like(y)], create_graph=True)
        del y
        dxs = [
            cupytorch.astensor(
                upd.send(
                    (
                        y_cpu[i],
                        cupytorch.asarray(gxs[i].detach()),
                    )
                )
            )
            for i, upd in enumerate(upds)
        ]
        hxdxs = torch.autograd.grad(gxs, xs, dxs)
        for i, upd in enumerate(upds):
            # Because of a bug in PyTorch, the line below cannot
            # `except StopIteration: pass`, which will cause
            # `SystemError: error return without exception set`.
            # Instead, `update` has a redundant `yield`.
            upd.send(cupytorch.asarray(hxdxs[i]))


class Trainer:
    best_y: float

    def __init__(self, *, fmax: float = 0.0, steps: int):
        self.max_iters = steps
        self.target_max_gx = fmax
        self.n_iters = -1  # observes initial values at the first iter

    def status_str(self) -> str:
        ss = [f"{self.n_iters:4} / {self.max_iters} iter"]
        if self.n_iters >= 0:
            y_diff = self.last_y - self.y0
            best = self.best_y - self.y0
            if y_diff == best:
                best_str = f"(best){' ':9}"
            else:
                best_str = f"(best: {best:7.4f})"
            ss.append(f"E - E_0 = {y_diff:7.4f} {best_str}")
            ss.append(f"max force = {self.last_max_gx:7.4f}")
        return ", ".join(ss)

    def _update_min(self, y: float, x: cupy.ndarray) -> None:
        if self.n_iters == 0 or y < self.best_y:
            self.best_x = x.copy()
            self.best_y = y

    def observe(self, *, x: cupy.ndarray, y: float, gx: cupy.ndarray) -> None:
        self.n_iters += 1
        if self.n_iters == 0:
            self.y0 = y
        self._update_min(y, x)
        self.last_y = y
        self.last_max_gx = cupy.asnumpy(cupy.sqrt(cupy.square(gx).sum(1).max()))[()]

    def is_done(self) -> bool:
        if self.n_iters <= 0:
            return False
        return self.last_max_gx < self.target_max_gx or self.n_iters >= self.max_iters


def _base_job(
    mol: CuPyMol,
    config: _OptConfig,
) -> Generator[Query, None, BaseOutput]:
    trainer = Trainer(steps=config.steps, fmax=config.fmax)
    logger.info(trainer.status_str())
    size = len(mol.atomic_numbers)
    priority = float("inf")
    opt = ScaledBFGS(
        mol,
        observe_fn=trainer.observe,
        lr=config.lr,
        maxstep=config.maxstep,
    )

    while True:
        yield Query(opt, size=size, priority=priority)
        logger.info(trainer.status_str())
        if trainer.is_done():
            break
        priority = trainer.last_max_gx
    mol.positions = trainer.best_x
    energy = trainer.best_y
    energy_before = trainer.y0
    info = {
        "energy": energy,
        "energy_before": energy_before,
        "energy_diff": energy - energy_before,
    }
    return info


ExecutorJob = Generator[Query, None, None]


class Executor:
    def __init__(
        self,
        estimator: "TeaNetEstimator",
        *,
        # optimizer config
        lr: float = _OptConfig.lr,
        maxstep: float = _OptConfig.maxstep,
        steps: int = _OptConfig.steps,
        fmax: float = _OptConfig.fmax,
        # schedule config
        capacity: int = 600,
        fetch_capacity: int = 900,
    ):
        net = MyTeaNet(estimator.model)  # type: ignore
        torch_device = net.device
        assert torch_device.type == "cuda"
        self._net = net
        self._cupy_device = cupy.cuda.Device(torch_device.index)
        self._config = _OptConfig(lr=lr, maxstep=maxstep, steps=steps, fmax=fmax)
        self._scheduler_kwargs = dict(capacity=capacity, fetch_capacity=fetch_capacity)

    def map(
        self,
        func: UserFunc[UserInput, UserOutput],
        iterable: Iterable[UserInput],
    ) -> List[UserOutput]:
        with self._cupy_device:
            optimize = functools.partial(_base_job, config=self._config)
            results: List[UserOutput] = []

            def job_iter() -> Iterator[ExecutorJob]:
                for i, x in enumerate(iterable):
                    results.append(None)  # type: ignore

                    def job(i: int = i, x: UserInput = x) -> ExecutorJob:
                        results[i] = yield from func(x, optimize=optimize)

                    yield job()

            scheduler = Scheduler(
                job_iter(),
                **self._scheduler_kwargs,
            )
            for batch in scheduler:
                ScaledBFGS.batch_update(batch, self._net.forward)
            return results
