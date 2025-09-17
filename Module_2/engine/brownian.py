# sde/brownian.py
import numpy as np

def brownian_bridge_increments(z: np.ndarray, T: int, dt: float) -> np.ndarray:
    """
    Map i.i.d. N(0,1) to Brownian path increments with Brownian bridge ordering.
    z shape: (N, T, dimW) i.i.d. normals.
    returns dW shape: (T, N, dimW), each increment ~ N(0, dt) but time-ordered with low effective dimension.
    """
    # Simple bridge: reorder time indices by dyadic partition to reduce effective dimension.

    # dyadic ordering
    order_lst = []
    def rec(a, b):
        if b - a <= 1:
            if a < T: order_lst.append(a)
            return
        m = (a + b) // 2
        if m < T: order_lst.append(m)
        rec(a, m)
        rec(m, b)
    rec(0, T)
    order = np.array(order_lst[:T], dtype=int)

    # assign normals in bridge order, then permute back to chronological
    dW = z[:, :T, :] * np.sqrt(dt)                   # (N, T, dimW)
    dW_bridge = np.zeros_like(dW)
    dW_bridge[:, order, :] = dW
    # transpose to (T, N, dimW) for simulators
    return np.transpose(dW_bridge, (1, 0, 2))
