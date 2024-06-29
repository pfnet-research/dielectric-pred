from typing import Dict, List, Optional, Union

import dacite
import numpy as np
from progressbar import ProgressBar

from pfp.nn.estimator_base import BaseEstimator, EstimatorSystem
from pfp.utils.atoms import atomic_symbol
from pfp.utils.errors import PFPError
from pfp.utils.messages import MessageEnum


def calculate(
    estimator: BaseEstimator,
    data: List[Dict[str, Optional[Union[float, Dict[str, int], List[str], np.ndarray, List[MessageEnum]]]]],
    b_size: int,
    calc: bool = True,
    show_progress: bool = True,
) -> None:
    n_data = len(data)
    n_batch = ((n_data - 1) // b_size) + 1
    data_input: List[EstimatorSystem] = []
    for d in data:
        assert isinstance(d["atomic_numbers"], np.ndarray)
        assert isinstance(d["coordinates"], np.ndarray)
        if d["cell"] is None:
            cell_modfiy = 100.0 * np.eyes(3, dtype=np.float32)
        else:
            assert isinstance(d["cell"], np.ndarray)
            cell_modify = d["cell"]
        d["properties"] = ["forces"]
        d["pbc"] = None if d["cell"] is None else np.ones((3,))

        d_convert = dacite.from_dict(data_class=EstimatorSystem, data=d)
        data_input.append(d_convert)

    it = range(n_batch)
    if show_progress:
        it = ProgressBar()(it)
    for ib in it:
        istart = ib * b_size
        if istart > n_data - 1:
            break
        iend = min((ib + 1) * b_size, n_data)

        if calc:
            results = estimator.batch_estimate(data_input[istart:iend])

            for i, r in enumerate(results):
                assert not isinstance(r, PFPError)
                data[i + istart]["p_energies"] = r["energy"]
                data[i + istart]["p_forces"] = r["forces"]
                data[i + istart]["p_charges"] = r["charges"]

        for i in range(istart, iend):
            cell = data[i]["cell"]
            if cell is None:
                volume = None
            else:
                assert isinstance(cell, np.ndarray)
                volume = float(np.sum(cell[0, :] * np.cross(cell[1, :], cell[2, :])))
            data[i]["volume"] = volume

    return
