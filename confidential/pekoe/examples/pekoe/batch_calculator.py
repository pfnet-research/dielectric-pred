import time
from typing import List

from ase import units
from ase.build import bulk, fcc111
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS, FIRE

from pekoe.calculators.batch_calculator import BatchCalculator, DynProcess
from pekoe.nn.models import DEFAULT_MODEL
from pekoe.nn.models.model_builder import build_estimator

if __name__ == "__main__":

    n_jobs = 4

    # prepare for batch calculation
    estimator = build_estimator(DEFAULT_MODEL, device="cuda:0")
    batch_calculator = BatchCalculator(estimator, n_jobs=n_jobs)

    process_list: List[DynProcess] = list()

    # 1st task
    def test_attach():
        time.sleep(0.1)
        print("This info comes from attached function.")

    atoms1 = bulk("Si") * (2, 2, 2)
    atoms1.set_calculator(batch_calculator)
    dyn1 = VelocityVerlet(atoms1, 1.0 * units.fs, trajectory="md1.traj")
    dyn1.attach(test_attach)
    process_list.append(DynProcess(target=dyn1.run, args=(8,)))

    # 2nd task
    atoms2 = bulk("Cu") * (2, 2, 2)
    atoms2.set_calculator(batch_calculator)
    bfgs = BFGS(atoms2, logfile="bfgs.log")
    process_list.append(
        DynProcess(target=bfgs.run, args=(), kwargs={"steps": 100, "fmax": 0.0001})
    )

    # 3rd task
    atoms3 = bulk("Al") * (2, 2, 2)
    atoms3.set_calculator(batch_calculator)
    fire = FIRE(atoms3, logfile="fire.log", force_consistent=False)
    process_list.append(DynProcess(target=fire.run, args=(), kwargs={"steps": 100, "fmax": 0.01}))

    # 4th task
    atoms4 = bulk("C") * (5, 5, 5)
    atoms4.set_calculator(batch_calculator)
    dyn4 = Langevin(atoms4, 2.0 * units.fs, temperature_K=300.0, friction=0.1)
    process_list.append(DynProcess(target=dyn4.run, args=(40,)))

    # 5th task
    atoms5 = bulk("Fe") * (2, 2, 2)
    atoms5.set_calculator(batch_calculator)
    MaxwellBoltzmannDistribution(atoms5, temperature_K=200.0)
    dyn5 = NPTBerendsen(
        atoms5,
        1.0 * units.fs,
        temperature_K=200.0,
        taup=500 * units.fs,
        pressure_au=1.01325 * (1e5 * units.Pascal),
        compressibility_au=4.57e-5 / (1e5 * units.Pascal),
    )
    process_list.append(DynProcess(target=dyn5.run, args=(30,)))

    # 6th task
    atoms6 = fcc111("Al", (2, 2, 3), vacuum=10.0)
    atoms6.set_pbc(True)
    atoms6.set_calculator(batch_calculator)

    def script_in_process6(atoms6):
        MaxwellBoltzmannDistribution(atoms6, temperature_K=300.0)
        dyn6 = NVTBerendsen(atoms6, 1.0 * units.fs, temperature_K=300.0, taut=500 * units.fs)
        dyn6.run(5)
        bfgs = BFGS(atoms6, logfile="bfgs6.log")
        bfgs.run(fmax=0.01, steps=20)
        dyn6 = Langevin(atoms6, 1.0 * units.fs, temperature_K=300.0, friction=0.01)
        dyn6.run(7)
        fire = FIRE(atoms6, logfile="fire6.log")
        fire.run(fmax=0.1, steps=100)

    process_list.append(DynProcess(target=script_in_process6, args=(atoms6,)))

    # 7th task
    atoms7 = fcc111("Cu", (2, 3, 3), vacuum=15.0)
    atoms7.set_pbc(True)
    atoms7.set_calculator(batch_calculator)
    process_list.append(DynProcess(target=atoms7.get_forces, args=()))

    for process in process_list:
        process.start()

    try:
        batch_calculator.run_calculate()
    except Exception as e:
        for process in process_list:
            process.kill()
        raise e

    for process in process_list:
        process.join()

    print("Finish")
