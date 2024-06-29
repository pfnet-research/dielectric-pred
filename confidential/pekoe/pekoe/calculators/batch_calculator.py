import ctypes
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize.optimize import Dynamics
from torch import multiprocessing as torch_multiprocessing

from pekoe.nn.estimator_base import EstimatorSystem
from pekoe.nn.models.teanet_estimator import TeaNetEstimator
from pfp.utils.errors import PFPError

"""
Batch calculator can be used based on python multiprocessing.
To be more specific, it allow us to run several ASE.Dynamics jobs from different process simultaneously.

Please be attention, after all the multiprocessing.Process started,
we need to run calculator from mainprocess, like :

    batch_calculator.run_calculate()

Otherwise the calculation will be stucked.
"""


class BatchCalculator(Calculator):  # type: ignore
    implemented_properties = ["energy", "forces", "stress", "charge"]
    name = "BatchCalculator"
    parameters: Dict[Any, Any] = {}

    def __init__(self, estimator: TeaNetEstimator, n_jobs: int = 2):
        super(BatchCalculator, self).__init__()

        # Using PFPCalculatorExtension to perform batch calculation
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.num_atoms = torch_multiprocessing.Value("i", 0)

        # Using torch_multiprocessing.Queue to passing information between subprocess and mainprocess
        self.queue: torch_multiprocessing.Queue[
            Tuple[int, EstimatorSystem]
        ] = torch_multiprocessing.Queue()
        self.queue_results: torch_multiprocessing.Queue[
            Dict[int, Any]
        ] = torch_multiprocessing.Queue()

        # Using torch_multiprocessing.Event to synchronize each dynamic process and calculator
        self.event_que = torch_multiprocessing.Event()
        self.event_que.set()
        self.event_calc = torch_multiprocessing.Event()
        self.event_update = torch_multiprocessing.Event()

        self.lock = torch_multiprocessing.Lock()

        # Using sema to control the maximum number of runing calculations
        self.sema = torch_multiprocessing.Semaphore(n_jobs)

        # create share memory arries. To save whether the atoms is alive.
        self.shm_alive = torch_multiprocessing.Array(ctypes.c_bool, 1000)
        self.alive = np.frombuffer(self.shm_alive.get_obj(), dtype=bool)
        self.alive[:] = 0

        # the atoms last calculated
        self.atoms = None

    def _bind_atoms(self, atoms: Atoms) -> None:
        """
        Binding atoms with calculator, set the ID of each atoms.
        """
        atoms.BATCH_CALCULATOR_ID = self.num_atoms.value
        self.num_atoms.value += 1

    def calculate(
        self,
        atoms: Atoms,
        properties: Optional[List[str]] = None,
        system_changes: List[str] = all_changes,
    ) -> None:
        """
        Run batch calculations.

        This function must be called from subprocess (multiprocessing.Process).
        """
        # Using semaphore to avoid number of inquiry exceed n_jobs
        self.sema.acquire()

        # set the current atom to alive
        if not hasattr(atoms, "BATCH_CALCULATOR_ID"):
            self._bind_atoms(atoms)

        self.toggle_alive(atoms.BATCH_CALCULATOR_ID, True)

        super(BatchCalculator, self).calculate(atoms, properties, system_changes)

        self.event_que.wait()
        estimator_inputs = EstimatorSystem(
            properties=["energy"],
            pbc=atoms.get_pbc().astype(np.uint8),
            atomic_numbers=atoms.get_atomic_numbers(),
            coordinates=atoms.get_positions(),
            cell=atoms.get_cell()[:],
        )
        self.queue.put((atoms.BATCH_CALCULATOR_ID, estimator_inputs))

        with self.lock:
            if self.queue.qsize() == min(self.n_jobs, self.alive.sum()):
                self.event_que.clear()
                self.event_calc.set()

        # wait until the batch calcuation in self.run_calculate finished.
        self.event_update.wait()

        try:
            # update the results
            self._update_results(atoms)
        finally:
            self.event_update.clear()
            self.event_que.set()
            self.sema.release()

    def _update_results(self, atoms: Atoms) -> None:
        """
        This function is called from subprocess.
        """
        # read results from queue and save to self.results
        ind = atoms.BATCH_CALCULATOR_ID
        results: Dict[int, Any] = self.queue_results.get()
        if ind not in results.keys():
            print(results.keys(), ind)

        self.results = results[ind]

        if "virial" in self.results and np.all(atoms.get_pbc()):
            self.results["stress"] = self.results["virial"] / atoms.get_volume()

    def run_calculate(self) -> None:
        """
        This function is called from mainprocess.

        When this function is runing in mainprocess, it constantly wait the signal (self.event_calc) to
        start the batch calculation.
        The calculation will be terminated when all the binding atoms are not in alive status.
        """
        # Using this function in mainprocess.
        # Calling CUDA from subprocess might cause error.

        while True:
            self.event_calc.wait()
            # print("    Start Calculation ")

            # get atoms from queue
            input_list: List[EstimatorSystem] = list()
            atoms_id_list: List[int] = list()
            n_atoms = self.queue.qsize()

            for _ in range(n_atoms):
                atoms_id, estimator_inputs = self.queue.get()
                input_list.append(estimator_inputs)
                atoms_id_list.append(atoms_id)

            # run batch calculation
            if n_atoms > 0:
                estimator_results = self.estimator.batch_estimate(input_list)
                results: Dict[int, Any] = dict()
                for i, result in enumerate(estimator_results):
                    assert not isinstance(result, PFPError)
                    results[atoms_id_list[i]] = result
            else:
                results = {}

            self.event_calc.clear()

            # send results
            for _ in range(n_atoms):
                self.queue_results.put(results)

            # allows to update results
            self.event_update.set()

            if self.alive.sum() == 0:
                break

    def toggle_alive(self, batch_calculator_id: int, flag: bool) -> None:
        with self.lock:
            if flag:
                self.alive[batch_calculator_id] = True
            else:
                self.alive[batch_calculator_id] = False
                if self.alive.sum() < self.n_jobs:
                    if self.queue.qsize() == min(self.n_jobs, self.alive.sum()):
                        self.event_calc.set()


class _Dyn:
    def __init__(self, func) -> None:  # type: ignore
        self.func = func

    def __call__(self, *args, **kwargs) -> None:  # type: ignore
        if hasattr(self.func, "__self__"):
            if isinstance(self.func.__self__, Dynamics):
                dyn = self.func.__self__
                atoms = dyn.atoms
            elif isinstance(self.func.__self__, Atoms):
                atoms = self.func.__self__
            else:
                raise TypeError("The DynProcess can accept Dynamics.run or Atoms")
        else:
            assert isinstance(args[0], Atoms)
            atoms = args[0]
        calc = atoms.calc
        assert isinstance(calc, BatchCalculator)
        if not hasattr(atoms, "BATCH_CALCULATOR_ID"):
            calc._bind_atoms(atoms)
        calc.toggle_alive(atoms.BATCH_CALCULATOR_ID, True)

        try:
            self.func(*args, **kwargs)
        except Exception as e:
            print(
                "The process ",
                torch_multiprocessing.current_process(),
                " terminated due to internal error",
            )
            raise e

        calc.toggle_alive(atoms.BATCH_CALCULATOR_ID, False)


class DynProcess(torch_multiprocessing.Process):  # type: ignore
    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        super(DynProcess, self).__init__(*args, **kwargs)
        self._target = _Dyn(self._target)  # type: ignore
