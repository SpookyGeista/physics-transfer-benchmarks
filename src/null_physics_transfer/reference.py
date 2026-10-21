"""Hidden reference laws. Training and predictor modules MUST NOT import this file.

Fixture: 2 kg frictionless horizontal guide, cam compression d(t)+s*x,
force u-s*N+F_t. Standards solver integrates rigid mass. Contact pair is
collision-disabled; these are the only material forces. Fixed apparatus
receives the reaction -F_t+s*N (not integrated). No custom PhysX solver.
"""
import numpy as np
from .common import DT, VREG

class Reference:
    def __init__(self, world, n, initial_delta):
        self.world=world
        self.memory=np.column_stack([initial_delta, initial_delta, np.zeros(n)])

    def force(self, delta, rate, v, forced_normal=None, friction=True, dt=DT):
        contact=delta>0
        rate=np.where(contact,rate,0); delta=np.maximum(delta,0)
        z=self.memory
        if self.world=='sls':
            # Standard linear solid with nonlinear equilibrium spring.
            normal=np.maximum(0, 560*delta + 360*(delta-z[:,0]) + 1.7e6*delta**3 + 3.5*rate)
            mu=.34+.19*np.exp(-(v/.16)**2)+.055*np.abs(z[:,2])
            tangential=-normal*mu*np.tanh(v/VREG)-.35*v
            z[:,0] += (1-np.exp(-dt/.13))*(delta-z[:,0])
            z[:,2] += (1-np.exp(-dt/.055))*(v-z[:,2])
        elif self.world=='nonlinear':
            # Independent constitutive variant: two relaxation branches,
            # pressure-sensitive friction and saturating viscoelastic force.
            normal=np.maximum(0, 430*delta + 7.0*np.tanh((delta-z[:,0])/.012)
                + 190*(delta-z[:,1]) + 7.5e5*delta**3 + 6*rate)
            mu=.27+.21/(1+(v/.12)**2)+.07*np.tanh(normal/15)+.09*np.abs(z[:,2])
            tangential=-normal*mu*np.tanh(v/VREG)-.55*v
            z[:,0] += (1-np.exp(-dt/.045))*(delta-z[:,0])
            z[:,1] += (1-np.exp(-dt/.24))*(delta-z[:,1])
            z[:,2] += (1-np.exp(-dt/.11))*(v-z[:,2])
        else:
            raise ValueError(self.world)
        normal=np.where(contact,normal,0.)
        tangential=np.where(normal>0,tangential,0.)
        if forced_normal is not None:
            # Recompute friction at the independently controlled calibration load.
            if self.world=='sls':
                ratio=.34+.19*np.exp(-(v/.16)**2)+.055*np.abs(self._old_v)
                visc=.35
            else:
                ratio=.27+.21/(1+(v/.12)**2)+.07*np.tanh(forced_normal/15)+.09*np.abs(self._old_v)
                visc=.55
            normal=forced_normal
            tangential=-normal*ratio*np.tanh(v/VREG)-visc*v
        if not isinstance(friction,bool):
            tangential=np.where(friction,tangential,0.)
        elif not friction:
            tangential=np.zeros_like(v)
        return normal, tangential

    def observe_force(self, *args, **kwargs):
        self._old_v=self.memory[:,2].copy()
        return self.force(*args, **kwargs)
