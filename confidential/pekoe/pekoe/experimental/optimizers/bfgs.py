from typing import Optional

import numpy as np

# TODO: Union[np.ndarray, cupy.ndarray]
ndarray = np.ndarray


class BFGS:

    """BFGS

    Example:

    .. code-block::

        bfgs = BFGS(...)
        while True:
            y = f(x)
            gx = grad(y, x)
            x += bfgs.step(x, gx)

    Args:
        lr (float): Initial guess of the inverse of the Hessian.

    """

    def __init__(self, *, lr: float):
        self._lr = lr
        self._last_x: Optional[ndarray] = None
        self._last_gx: Optional[ndarray] = None

    def step(self, x: ndarray, gx: ndarray) -> ndarray:
        x_flat = x.ravel()
        gx_flat = gx.ravel()
        self._update(x_flat, gx_flat)
        dx_flat = self._get_step(gx_flat)
        dx = dx_flat.reshape(x.shape)
        return dx

    def _get_step(self, gx: ndarray) -> ndarray:
        dx: ndarray = -(self._inv_h @ gx)
        return dx

    def _update(self, x: ndarray, gx: ndarray) -> None:
        last_x = self._last_x
        last_gx = self._last_gx
        self._last_x = x.copy()
        self._last_gx = gx.copy()

        if last_x is None:
            assert x.ndim == 1
            self._inv_h = np.diag(np.full_like(x, self._lr))
            return

        s = x - last_x
        y = gx - last_gx

        inv_h = self._inv_h

        # TODO: What should be done if (0 <= s @ y < eps)?
        # no update (alpha == 0) if s @ y < 0
        alpha = np.fmax(0, np.reciprocal(s @ y))
        tmp = alpha * (inv_h @ y)
        coef_ss = alpha * (1 + y @ tmp)
        tmp2 = np.outer(tmp, s)
        tmp2 += tmp2.T
        tmp2 -= np.outer(coef_ss * s, s)
        inv_h -= tmp2

        self._inv_h = inv_h
