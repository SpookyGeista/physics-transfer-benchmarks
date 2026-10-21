"""Independent native MuJoCo implementation of the unchanged round-1 laws.

Evaluation-only: no primary reference, stimulus generator, predictor, optimizer,
or hardware imports. Fixed synthetic constants are independently expressed as in
the previously audited MuJoCo reference. They are not parameters available to a
learner. Maxwell stores branch forces and a dimensionless friction fraction.
"""
import time

import numpy as np

DT = .0025
MASS = 2.


class IndependentMaterial:
    """Return pre-update force, then advance each material state once."""
    def __init__(self, world, n, initial_delta):
        if world not in ("maxwell", "boucwen"):
            raise ValueError(world)
        if isinstance(n, bool) or int(n) != n or n < 1:
            raise ValueError("Positive integer batch required")
        self.world, self.n = world, int(n)
        self.previous_delta = np.maximum(self._array(initial_delta), 0).copy()
        self.branch_force = np.zeros((self.n, 3))
        self.stribeck_fraction = np.ones(self.n)
        self.z = np.tanh(self.previous_delta/.009)
        self.age = np.full(self.n, .008/.12)

    def _array(self, value):
        a = np.broadcast_to(np.asarray(value, dtype=float), (self.n,))
        if not np.isfinite(a).all():
            raise ValueError("Finite scalar or batch array required")
        return a

    def force(self, delta, rate, velocity, normal_override=None,
              friction_enabled=True, dt=DT):
        if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Positive finite timestep required")
        d, dr, vel = [self._array(x) for x in (delta, rate, velocity)]
        active = d > 0
        dplus = np.maximum(d, 0)
        dr = np.where(active, dr, 0)
        speed = np.abs(vel)
        if self.world == "maxwell":
            branch_now = self.branch_force + (dplus-self.previous_delta)[:, None]*np.array([210., 125., 80.])
            raw_load = 630*dplus + branch_now.sum(1) + 2.2*dr
        else:
            raw_load = 680*dplus + 4.5*self.z + 1.8*dr
        load = raw_load if normal_override is None else self._array(normal_override)
        normal = np.where(active, np.maximum(load, 0), 0)
        loaded = active & (normal > 0)
        if self.world == "maxwell":
            goal_fraction = np.exp(-(speed/.11)**1.4)
            mu = (.29 + .24*(.65*goal_fraction + .35*self.stribeck_fraction))
            mu *= np.exp(-.055*np.log((normal+3)/18))
            viscous = .35
        else:
            mu = .40 + .018*np.log(np.sqrt(vel*vel + .002**2)/.12)
            mu = np.maximum(mu + .026*np.log(self.age/(.008/.12)), 0)
            viscous = .40
        friction = np.where(loaded & np.broadcast_to(friction_enabled, (self.n,)),
                            -(normal*mu*np.tanh(vel/.04) + viscous*vel), 0)
        if self.world == "maxwell":
            self.branch_force = branch_now*np.exp(-dt/np.array([.025, .17, .80]))
            self.previous_delta = dplus.copy()
            decay = np.exp(-dt/(.05 + .001*normal))
            goal = np.where(loaded, goal_fraction, 1.)
            self.stribeck_fraction = goal + (self.stribeck_fraction-goal)*decay
        else:
            scaled_rate = dr/.009
            count = max(1, int(np.ceil(np.max(np.abs(scaled_rate))*dt/.05)))
            h = dt/count

            def derivative(z):
                return scaled_rate*(1-.35*z*z) - .65*np.abs(scaled_rate)*np.abs(z)*z

            z = self.z.copy()
            for _ in range(count):
                a = h*derivative(z)
                b = h*derivative(z+a/2)
                c = h*derivative(z+b/2)
                e = h*derivative(z+c)
                z += (a+2*b+2*c+e)/6
            self.z = np.where(active, z, self.z*np.exp(-dt/.12))
            inverse_time = speed/.008
            growth = np.divide(-np.expm1(-inverse_time*dt), inverse_time,
                               out=np.full(self.n, dt), where=inverse_time > 0)
            next_age = self.age*np.exp(-inverse_time*dt) + growth
            recovered = .008/.12 + (self.age-.008/.12)*np.exp(-dt/.20)
            self.age = np.where(loaded, next_age, recovered)
        return normal, friction


class IndependentComposite:
    """Two independent states, coupled only through the active normal force."""
    def __init__(self, normal_world, friction_world, n, initial_delta=.02):
        self.n = n
        self.normal = IndependentMaterial(normal_world, n, initial_delta)
        self.friction = IndependentMaterial(friction_world, n, initial_delta)
        self.updates = 0

    def force(self, delta, rate, velocity, dt=DT):
        n, _ = self.normal.force(delta, rate, np.zeros(self.n),
                                 friction_enabled=False, dt=dt)
        _, ft = self.friction.force(delta, rate, velocity, normal_override=n, dt=dt)
        self.updates += 1
        return n, ft


def cam(position, amplitude):
    """Independent expression of the same bounded mechanical geometry."""
    value = np.tanh(np.asarray(position)/.04)
    return amplitude*value, amplitude*(1-value*value)/.04


def preparation(time_seconds, rows):
    duration = np.asarray([r['preparation_duration'] for r in rows])
    amplitude = np.asarray([r['preparation_sign']*r['preparation_amplitude'] for r in rows])
    phase = np.pi*time_seconds/duration
    active = (time_seconds > 0) & (time_seconds < duration)
    return (.02 + np.where(active, amplitude*np.sin(phase)**2, 0),
            np.where(active, amplitude*np.pi/duration*np.sin(2*phase), 0))


def compression(time_seconds, rows):
    magnitude = np.asarray([r['wave_amplitude'] for r in rows])
    omega = 2*np.pi*np.asarray([r['wave_frequency'] for r in rows])
    angle = omega*time_seconds + np.asarray([r['phase'] for r in rows])
    return .02 + magnitude*np.sin(angle), magnitude*omega*np.cos(angle)


def make_model(n, dt=DT):
    import mujoco
    if isinstance(n, bool) or int(n) != n or n < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Positive batch and timestep required")
    bodies = ''.join(f'''<body name="slider_{i}" pos="0 {i*.3:.17g} .1">
       <joint name="guide_{i}" type="slide" axis="1 0 0" damping="0" frictionloss="0" armature="0"/>
       <geom type="box" size=".06 .05 .04" mass="{MASS:.17g}" contype="0" conaffinity="0"/>
       </body>''' for i in range(n))
    return mujoco.MjModel.from_xml_string(f'''<mujoco model="round1_independent_cartesian_replay">
       <option timestep="{dt:.17g}" gravity="0 0 0" integrator="Euler" solver="Newton" iterations="100" tolerance="1e-12"/>
       <worldbody>{bodies}</worldbody></mujoco>''')


def replay(normal_world, friction_world, rows, force_commands, substeps=1,
           dt=DT, warmup_seconds=4., deadline=None):
    """Native replay at finer h; command u remains frozen on the coarse grid.

    Every slider begins x=v=0, material preparation is analytically evaluated
    through the full prefix, and the native slider is held by zero net applied
    force during that preparation. Only the external d waveform is prescribed;
    moving compression/rate use each native slider's current x/v and cam slope.
    No force correction or reference-state injection is used after release.
    Returned `state`/`next_state` align with coarse pre/post-step observations.
    """
    import mujoco
    if isinstance(substeps, bool) or substeps not in (1, 2, 4):
        raise ValueError("Only the declared 1, 2, 4 substeps are supported")
    if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Positive finite coarse timestep required")
    u = np.asarray(force_commands, dtype=float)
    n = len(rows)
    if u.ndim != 2 or u.shape[0] != n or not np.isfinite(u).all():
        raise ValueError("Finite episode-by-coarse-time force commands required")
    if not np.isfinite(warmup_seconds) or warmup_seconds < 0:
        raise ValueError("Nonnegative finite preparation duration required")
    warm_steps = round(warmup_seconds/dt)
    if not np.isclose(warm_steps*dt, warmup_seconds, atol=1e-14, rtol=0):
        raise ValueError("Preparation must align with the coarse grid")
    h = dt/substeps
    model = make_model(n, h)
    native = mujoco.MjData(model)
    mujoco.mj_forward(model, native)
    material = IndependentComposite(normal_world, friction_world, n)
    amplitude = np.asarray([r['amplitude'] for r in rows])
    out = {key: np.empty((n, u.shape[1], 2 if key in ('state', 'next_state') else 1))
           for key in ('state', 'next_state', 'normal', 'friction', 'delta', 'compression_rate')}
    out = {key: value if value.shape[-1] == 2 else value[..., 0] for key, value in out.items()}
    warm = {key: np.empty((n, warm_steps)) for key in ('normal', 'friction', 'delta', 'rate')}
    peak_warm_motion = 0.

    def check_budget(step):
        if deadline is not None and step % 100 == 0 and time.monotonic() > deadline:
            raise TimeoutError("Declared backend wall-time limit reached; no result substituted")

    for j in range(warm_steps*substeps):
        check_budget(j)
        d, rate = preparation(j*h, rows)
        normal, friction = material.force(d, rate, np.zeros(n), dt=h)
        if j % substeps == 0:
            t = j//substeps
            for key, value in [('normal', normal), ('friction', friction), ('delta', d), ('rate', rate)]:
                warm[key][:, t] = value
        native.qfrc_applied[:] = 0
        mujoco.mj_step(model, native)
        peak_warm_motion = max(peak_warm_motion, float(np.max(np.abs(native.qpos))),
                               float(np.max(np.abs(native.qvel))))
    for t in range(u.shape[1]):
        check_budget(t)
        out['state'][:, t] = np.column_stack([native.qpos, native.qvel])
        for sub in range(substeps):
            moment = (t*substeps+sub)*h
            d, rate = compression(moment, rows)
            gap, slope = cam(native.qpos, amplitude)
            delta, dr = d+gap, rate+slope*native.qvel
            normal, friction = material.force(delta, dr, native.qvel, dt=h)
            if sub == 0:
                for key, value in [('normal', normal), ('friction', friction), ('delta', delta), ('compression_rate', dr)]:
                    out[key][:, t] = value
            native.qfrc_applied[:] = u[:, t] - slope*normal + friction
            mujoco.mj_step(model, native)
        out['next_state'][:, t] = np.column_stack([native.qpos, native.qvel])
    if not all(np.isfinite(value).all() for value in [*out.values(), *warm.values()]):
        raise FloatingPointError("Independent native replay is nonfinite; no clipping applied")
    out.update({'warm_'+key: value for key, value in warm.items()})
    return out, dict(backend='native MuJoCo independent slide joints',
                     mujoco_version=mujoco.__version__, dt_observation=dt,
                     dt_integrator=h, substeps=substeps, episodes=n,
                     warmup_seconds=warmup_seconds, warm_motion_max=peak_warm_motion,
                     material_updates=material.updates, native_collisions=False,
                     force_hold='recorded u held constant per original coarse interval',
                     compression='analytic prescribed waveform plus actual native cam position/rate',
                     preparation='analytic pulse and hold, x=v=0; full prefix separately integrated at each h')
