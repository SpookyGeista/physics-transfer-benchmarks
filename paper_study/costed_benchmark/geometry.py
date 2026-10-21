"""Public kinematic coupling functions; no material or fitted-model knowledge."""
import numpy as np

GEOMETRY_NAMES = {0: "tanh", 1: "sin", 2: "bump"}
LENGTH_M = .012


def cam(x, amplitude, geometry):
    """Return g(x), dg/dx; integer IDs broadcast with x and amplitude.

    g=A*tanh(z), A*sin(z), or A*z²/(1+z²), where z=x/.012.
    Geometry is a known assembly property available to every predictor.
    """
    x, amplitude, geometry = np.broadcast_arrays(np.asarray(x, dtype=float),
                                                np.asarray(amplitude, dtype=float), np.asarray(geometry))
    if not np.isfinite(x).all() or not np.isfinite(amplitude).all():
        raise ValueError("Finite geometry inputs required")
    if geometry.dtype.kind not in "iu" or not np.isin(geometry, list(GEOMETRY_NAMES)).all():
        raise ValueError("geometry must use integer IDs 0=tanh, 1=sin, 2=bump")
    z = x / LENGTH_M
    g = np.empty_like(x)
    slope = np.empty_like(x)
    for identity in GEOMETRY_NAMES:
        selected = geometry == identity
        zz, aa = z[selected], amplitude[selected]
        if identity == 0:
            tangent = np.tanh(zz)
            g[selected], slope[selected] = aa * tangent, aa / LENGTH_M * (1-tangent**2)
        elif identity == 1:
            g[selected], slope[selected] = aa * np.sin(zz), aa / LENGTH_M * np.cos(zz)
        else:
            denominator = 1 + zz**2
            g[selected], slope[selected] = aa * zz**2 / denominator, 2*aa/LENGTH_M*zz/denominator**2
    return g, slope
