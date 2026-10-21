"""Passive semi-implicit guided-mass step with a monotone stop-force law.

At fixed current N/slope and held external force, solve the velocity using the
stop law's exact force integral over the ACTUAL semi-implicit displacement.
The caller supplies a pure force-average function; this module imports neither
a reference nor a learned material. A native engine may reproduce the resulting
force step; its float precision residual must be audited separately.
"""
import numpy as np


def solve_velocity(v, u, slope, normal, capacity, force_average, *, dt=.0025, mass=2., iterations=40):
    """Find vn = v + dt/m*(u-slope*N+mean_Ft(vn*dt)).

    force_average receives trial interval displacement (meters). Its output
    must lie in [-capacity,+capacity] and be nonincreasing in displacement, as
    for parallel positive-stiffness stops with fixed load. These are declared
    assumptions, not checked by sampling here. Forty bisections bound the
    velocity error by the initial bracket width / 2**40.
    """
    v,u,slope,normal,capacity=np.broadcast_arrays(*[np.asarray(a,dtype=float) for a in (v,u,slope,normal,capacity)])
    if any(not np.isfinite(a).all() for a in (v,u,slope,normal,capacity)):
        raise ValueError("Finite coupling inputs required")
    if (np.any(normal<0) or np.any(capacity<0) or not np.isscalar(dt) or not np.isfinite(dt)
            or dt<=0 or not np.isscalar(mass) or not np.isfinite(mass) or mass<=0):
        raise ValueError("Nonnegative load/capacity and positive step/mass required")
    if (not np.isscalar(iterations) or isinstance(iterations,bool) or not np.isfinite(iterations)
            or int(iterations)!=iterations or iterations<1):
        raise ValueError("Positive integer bisection count required")
    iterations=int(iterations)
    def checked_force(displacement):
        try:
            force=np.broadcast_to(np.asarray(force_average(displacement),dtype=float),v.shape)
        except ValueError as exc:
            raise ValueError("Force-average output must broadcast to the input shape") from exc
        if not np.isfinite(force).all():
            raise FloatingPointError("Nonfinite force-average output")
        if np.any(np.abs(force)>capacity+32*np.finfo(float).eps*np.maximum(capacity,1.)):
            raise ValueError("Force-average output exceeds the declared capacity")
        return force
    free=v+dt*(u-slope*normal)/mass
    lo,hi=free-dt*capacity/mass,free+dt*capacity/mass
    for _ in range(iterations):
        mid=(lo+hi)/2
        force=checked_force(mid*dt)
        residual=mid-free-dt*force/mass
        lo=np.where(residual<0,mid,lo)
        hi=np.where(residual>=0,mid,hi)
    velocity=(lo+hi)/2
    force=checked_force(velocity*dt)
    residual=mass*(velocity-v)/dt-(u-slope*normal+force)
    if not np.isfinite(force).all() or not np.isfinite(residual).all():
        raise FloatingPointError("Nonfinite implicit force solution")
    return velocity, force, residual
