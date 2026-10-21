"""Identified persistent material states with an explicitly privileged known family.

Production code imports no simulation/reference module and reads no true parameters.
Only measured isolated sequences identify coefficients. Constitutive family, n=2
Bouc-Wen preparation form, mass-independent SI inputs and vreg are known priors.
Opening relaxation is not identifiable from closed-contact data and is nominal.
"""
import json
import time
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import least_squares, nnls

VREG = .04


@njit(cache=True)
def _maxwell_normal_basis(d, rate, tau, dt):
    out = np.empty((d.shape[0], d.shape[1], 5))
    for i in range(d.shape[0]):
        h = np.full(3, max(d[i, 0], 0.))
        for t in range(d.shape[1]):
            x = max(d[i, t], 0.)
            out[i, t, 0] = x / .02
            out[i, t, 1] = (rate[i, t] if x > 0 else 0.) / .1
            for j in range(3):
                out[i, t, j+2] = (x-h[j]) / .02
                h[j] += -np.expm1(-dt/tau[j]) * (x-h[j])
    return out


@njit(cache=True)
def _bw_dz(z, r, beta, gamma):
    return r - beta * abs(r) * abs(z) * z - gamma * r * z*z


@njit(cache=True)
def _bw_advance_scalar(z, rate, dt, dy, beta, gamma):
    r = rate/dy
    count = max(1, int(np.ceil(abs(r)*dt/.05)))
    h = dt/count
    for _ in range(count):
        a = _bw_dz(z, r, beta, gamma)
        b = _bw_dz(z+.5*h*a, r, beta, gamma)
        c = _bw_dz(z+.5*h*b, r, beta, gamma)
        e = _bw_dz(z+h*c, r, beta, gamma)
        z += h*(a+2*b+2*c+e)/6
    return z


@njit(cache=True)
def _bw_normal_basis(d, rate, dy, beta, gamma, dt):
    out = np.empty((d.shape[0], d.shape[1], 3))
    for i in range(d.shape[0]):
        z = np.tanh(max(d[i, 0], 0.)/dy)
        for t in range(d.shape[1]):
            x = max(d[i, t], 0.)
            r = rate[i, t] if x > 0 else 0.
            out[i, t, 0] = x/.02
            out[i, t, 1] = r/.1
            out[i, t, 2] = z
            z = _bw_advance_scalar(z, r, dt, dy, beta, gamma)
    return out


@njit(cache=True)
def _maxwell_friction_basis(v, normal, p, dt):
    # Nonlinear p: speed, exponent, mixing, lag0, lag/load, offset, load exponent.
    vs, power, mixing, lag0, lagload, offset, loadpower = p
    out = np.empty((v.shape[0], v.shape[1], 3))
    for i in range(v.shape[0]):
        memory = 1.
        for t in range(v.shape[1]):
            speed = abs(v[i, t]); n = max(normal[i, t], 0.)
            e = np.exp(-(speed/vs)**power)
            mix = mixing*e+(1-mixing)*memory
            # 20 N is a declared gauge, not an identified physical constant.
            factor = ((n+offset)/20.)**loadpower
            drag = -n*np.tanh(v[i, t]/VREG)*factor
            out[i, t, 0] = drag
            out[i, t, 1] = drag*mix
            out[i, t, 2] = -v[i, t]
            target = e if n > 0 else 1.
            memory += -np.expm1(-dt/(lag0+lagload*n))*(target-memory)
    return out


@njit(cache=True)
def _bw_friction_basis(v, normal, slip, theta0, floor, dt):
    out = np.empty((v.shape[0], v.shape[1], 4))
    reference_speed = slip/theta0
    for i in range(v.shape[0]):
        age = theta0
        for t in range(v.shape[1]):
            speed = abs(v[i, t]); n = max(normal[i, t], 0.)
            drag = -n*np.tanh(v[i, t]/VREG)
            out[i, t, 0] = drag
            out[i, t, 1] = drag*np.log(np.sqrt(v[i, t]**2+floor**2)/reference_speed)
            out[i, t, 2] = drag*np.log(age/theta0)
            out[i, t, 3] = -v[i, t]
            inverse = speed/slip
            growth = -np.expm1(-dt*inverse)/inverse if inverse > 0 else dt
            age = age*np.exp(-dt*inverse)+growth
    return out


class StateID:
    """Pure causal stepping; every returned state owns independent arrays."""
    def __init__(self, world, normal, friction, metadata=None):
        if world not in ('maxwell', 'boucwen'):
            raise ValueError(world)
        self.world = world
        self.normal = normal
        self.friction = friction
        self.metadata = metadata or {}

    def initial(self, delta, v, normal_sensor):
        d, _, _ = np.broadcast_arrays(np.atleast_1d(delta), np.atleast_1d(v),
                                      np.atleast_1d(normal_sensor))
        if d.ndim != 1 or not np.isfinite(d).all():
            raise ValueError('initial signals must be finite one-dimensional arrays')
        if self.world == 'maxwell':
            return dict(relaxed=np.repeat(np.maximum(d, 0.)[:, None], 3, axis=1),
                        mix=np.ones(len(d)))
        return dict(z=np.tanh(np.maximum(d, 0.)/self.normal['dy']),
                    age=np.full(len(d), self.friction['theta0']))

    def step(self, delta, rate, v, state, normal_override=None, dt=.0025):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('positive finite dt required')
        d, r, velocity = [np.atleast_1d(np.asarray(x, dtype=float)) for x in (delta, rate, v)]
        d, r, velocity = np.broadcast_arrays(d, r, velocity)
        if d.ndim != 1 or not all(np.isfinite(x).all() for x in (d, r, velocity)):
            raise ValueError('finite one-dimensional signals required')
        active = d > 0; compression = np.maximum(d, 0.); r = np.where(active, r, 0.)
        q, p = self.normal, self.friction
        if self.world == 'maxwell':
            raw = q['k_eq']*compression + q['c']*r + np.sum(
                np.asarray(q['k_branches'])*(compression[:, None]-state['relaxed']), axis=1)
        else:
            raw = q['k']*compression+q['c']*r+q['F']*state['z']
        applied = raw if normal_override is None else np.broadcast_to(normal_override, d.shape)
        if not np.isfinite(applied).all():
            raise ValueError('finite normal load required')
        n = np.where(active, np.maximum(applied, 0.), 0.)
        loaded = active & (n > 0)
        if self.world == 'maxwell':
            steady = np.exp(-(np.abs(velocity)/p['vs'])**p['power'])
            mixture = p['mixing']*steady+(1-p['mixing'])*state['mix']
            mu = (p['mu_c']+p['delta_mu']*mixture)*((n+p['load_offset'])/20.)**p['load_exponent']
            lag = p['tau0']+p['tau_load']*n
            new = dict(relaxed=state['relaxed'] + (-np.expm1(-dt/np.asarray(q['tau_branches'])))[None, :]
                       * (compression[:, None]-state['relaxed']),
                       mix=state['mix']+(-np.expm1(-dt/lag))*(np.where(loaded, steady, 1.)-state['mix']))
        else:
            reference_speed = p['slip']/p['theta0']
            mu = np.maximum(p['mu0'] + p['a']*np.log(np.hypot(velocity, p['vfloor'])/reference_speed)
                            + p['b']*np.log(state['age']/p['theta0']), 0.)
            z = np.array([_bw_advance_scalar(z0, rr, dt, q['dy'], q['beta'], q['gamma'])
                          for z0, rr in zip(state['z'], r)])
            inverse = np.abs(velocity)/p['slip']
            growth = np.divide(-np.expm1(-dt*inverse), inverse,
                               out=np.full(len(d), dt), where=inverse > 0)
            age = state['age']*np.exp(-dt*inverse)+growth
            # Explicit nominal recovery; closed-contact identification makes no
            # claim about unobserved opening dynamics. Same assumption for both.
            relaxation = np.exp(-dt/.1)
            new = dict(z=np.where(active, z, state['z']*relaxation),
                       age=np.where(loaded, age, p['theta0']+(state['age']-p['theta0'])*relaxation))
        ft = np.where(loaded, -n*mu*np.tanh(velocity/VREG)-p['viscous']*velocity, 0.)
        return n, ft, new

    def save(self, path):
        Path(path).write_text(json.dumps(dict(world=self.world, normal=self.normal,
                            friction=self.friction, metadata=self.metadata), indent=2, allow_nan=False)+'\n')

    @classmethod
    def load(cls, path):
        stored = json.loads(Path(path).read_text())
        if stored.get('type') == 'composite':
            return CompositeStateID(StateID(**stored['normal_model']), StateID(**stored['friction_model']),
                                    stored['metadata'])
        return cls(**stored)


class CompositeStateID:
    """Locally selected known families; a flat array state supports MPC cloning.

    Selection uses isolated validation losses, not a material/world identifier.
    The selected normal output is the sole current load passed to friction.
    """
    def __init__(self, normal_model, friction_model, metadata=None):
        self.normal_model = normal_model
        self.friction_model = friction_model
        self.normal_family = normal_model.world
        self.friction_family = friction_model.world
        self.metadata = metadata or {}

    def initial(self, delta, v, normal_sensor):
        ns = self.normal_model.initial(delta, v, normal_sensor)
        fs = self.friction_model.initial(delta, v, normal_sensor)
        return {**{'normal__'+k: a for k, a in ns.items()},
                **{'friction__'+k: a for k, a in fs.items()}}

    def step(self, delta, rate, v, state, normal_override=None, dt=.0025):
        ns = {k[len('normal__'):]: a for k, a in state.items() if k.startswith('normal__')}
        fs = {k[len('friction__'):]: a for k, a in state.items() if k.startswith('friction__')}
        normal, _, next_n = self.normal_model.step(delta, rate, v, ns, normal_override=normal_override, dt=dt)
        _, friction, next_f = self.friction_model.step(delta, rate, v, fs, normal_override=normal, dt=dt)
        return normal, friction, {**{'normal__'+k: a for k, a in next_n.items()},
                                  **{'friction__'+k: a for k, a in next_f.items()}}

    def save(self, path):
        def model_dict(model):
            return dict(world=model.world, normal=model.normal, friction=model.friction, metadata=model.metadata)
        Path(path).write_text(json.dumps(dict(type='composite', normal_model=model_dict(self.normal_model),
                            friction_model=model_dict(self.friction_model), metadata=self.metadata),
                            indent=2, allow_nan=False)+'\n')

    @classmethod
    def load(cls, path):
        result = StateID.load(path)
        if not isinstance(result, cls):
            raise ValueError('stored model is not composite')
        return result


def _project(basis, labels):
    matrix = basis.reshape(-1, basis.shape[-1]); y = labels.ravel()
    scales = np.maximum(np.linalg.norm(matrix, axis=0), 1e-12)
    coef = nnls(matrix/scales, y, maxiter=10000)[0]/scales
    return coef, (matrix@coef).reshape(labels.shape)


class _BudgetReached(Exception):
    pass


def _fit_head(train, validation, basis_fn, starts, bounds, deadline, scale, max_nfev):
    """Optimize training error only; validation selects whole restart candidates."""
    candidates = []
    for start in starts:
        if candidates and time.perf_counter() >= deadline:
            break
        best = {'loss': float('inf')}; evaluations = 0
        def residual(q):
            nonlocal evaluations
            if evaluations and time.perf_counter() >= deadline:
                raise _BudgetReached
            basis = basis_fn(q, train)
            if not np.isfinite(basis).all():
                return np.full(train['target'].size, 1e6)
            coef, pred = _project(basis, train['target'])
            error = (pred-train['target']).ravel()/scale
            loss = float(np.mean(error*error)); evaluations += 1
            if loss < best['loss']:
                best.update(loss=loss, q=np.array(q, copy=True), coef=coef.copy())
            return error
        timed_out = False
        try:
            result = least_squares(residual, start, bounds=bounds, max_nfev=max_nfev,
                                   ftol=1e-10, xtol=1e-9, gtol=1e-9, diff_step=1e-4)
            status, message = int(result.status), str(result.message)
        except _BudgetReached:
            timed_out = True; status = -2; message = 'fixed wall-clock budget reached'
        if 'q' not in best:
            raise RuntimeError('no finite identified candidate')
        vb = basis_fn(best['q'], validation)
        vp = (vb.reshape(-1, vb.shape[-1])@best['coef']).reshape(validation['target'].shape)
        vrmse = float(np.sqrt(np.mean((vp-validation['target'])**2)))
        candidates.append(dict(q=best['q'], coef=best['coef'], validation_rmse=vrmse,
                               train_rmse=scale*np.sqrt(best['loss']), evaluations=evaluations,
                               status=status, message=message, timed_out=timed_out))
    selected = min(candidates, key=lambda x: x['validation_rmse'])
    return selected, [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in item.items()}
                       for item in candidates]


def fit_state_id(world, train, validation, dt=.0025, max_seconds=120., seed=0, restarts=3):
    """Fit measured isolated sequences.

    train/validation each contain normal={delta,rate,target} (B) and
    friction={velocity,normal,target} (A). All arrays shape [episode,time].
    Only whole isolated validation sequences select restart candidates.
    """
    if world not in ('maxwell', 'boucwen') or max_seconds <= 0 or restarts < 1:
        raise ValueError('invalid family, time budget or restart count')
    for split in (train, validation):
        for head in ('normal', 'friction'):
            arrays = split[head]
            shape = np.asarray(arrays['target']).shape
            assert len(shape) == 2 and min(shape) > 0
            assert all(np.asarray(x).shape == shape and np.isfinite(x).all() for x in arrays.values())
        if np.any(split['normal']['delta'] <= 0) or np.any(split['friction']['normal'] <= 0):
            raise ValueError('this fit identifies closed loaded contact only')
    started = time.perf_counter(); rng = np.random.default_rng(seed)
    if world == 'maxwell':
        lo = np.log([.003, .003, .003]); hi = np.log([5., 5., 5.])
        starts = [np.log([.01, .1, 1.])]
        starts += [np.clip(starts[0]+rng.normal(0, .65, 3), lo+.01, hi-.01) for _ in range(restarts-1)]
        def nb(q, x):
            return _maxwell_normal_basis(x['delta'], x['rate'], np.exp(q), dt)
    else:
        lo = np.log([.001, .05, .05]); hi = np.log([.1, 3., 3.])
        starts = [np.log([.02, .5, .5])]
        starts += [np.clip(starts[0]+rng.normal(0, .5, 3), lo+.01, hi-.01) for _ in range(restarts-1)]
        def nb(q, x):
            dy, beta, gamma = np.exp(q)
            return _bw_normal_basis(x['delta'], x['rate'], dy, beta, gamma, dt)
    normal_fit, nc = _fit_head(train['normal'], validation['normal'], nb, starts, (lo, hi),
                              started+max_seconds*.45, 20., 100)
    q, coef = normal_fit['q'], normal_fit['coef']
    if world == 'maxwell':
        order = np.argsort(q)
        normal = dict(k_eq=float(coef[0]/.02), c=float(coef[1]/.1),
                      k_branches=(coef[2:][order]/.02).tolist(), tau_branches=np.exp(q[order]).tolist())
        # q transform: log speed/power; direct mix; log lags/offset; direct exponent.
        lo = np.array([np.log(.005), np.log(.3), .001, np.log(.001), np.log(1e-6), np.log(.1), -.5])
        hi = np.array([np.log(1.), np.log(4.), .999, np.log(2.), np.log(.1), np.log(100.), .5])
        first = np.array([np.log(.08), np.log(2.), .5, np.log(.1), np.log(.002), np.log(10.), -.1])
        starts = [first]
        for _ in range(restarts-1):
            perturb = rng.normal(0, .5, 7); perturb[[2, 6]] *= .2
            starts.append(np.clip(first+perturb, lo+1e-6, hi-1e-6))
        def fb(q, x):
            p = np.array([np.exp(q[0]), np.exp(q[1]), q[2], np.exp(q[3]), np.exp(q[4]), np.exp(q[5]), q[6]])
            return _maxwell_friction_basis(x['velocity'], x['normal'], p, dt)
    else:
        normal = dict(k=float(coef[0]/.02), c=float(coef[1]/.1), F=float(coef[2]),
                      dy=float(np.exp(q[0])), beta=float(np.exp(q[1])), gamma=float(np.exp(q[2])))
        lo = np.log([.0001, .001, .0001]); hi = np.log([.3, 10., .05])
        starts = [np.log([.01, .1, .005])]
        starts += [np.clip(starts[0]+rng.normal(0, .5, 3), lo+.01, hi-.01) for _ in range(restarts-1)]
        def fb(q, x):
            slip, theta0, floor = np.exp(q)
            return _bw_friction_basis(x['velocity'], x['normal'], slip, theta0, floor, dt)
    friction_fit, fc = _fit_head(train['friction'], validation['friction'], fb, starts, (lo, hi),
                                started+max_seconds, 10., 150)
    q, coef = friction_fit['q'], friction_fit['coef']
    if world == 'maxwell':
        friction = dict(mu_c=float(coef[0]), delta_mu=float(coef[1]), viscous=float(coef[2]),
                        vs=float(np.exp(q[0])), power=float(np.exp(q[1])), mixing=float(q[2]),
                        tau0=float(np.exp(q[3])), tau_load=float(np.exp(q[4])),
                        load_offset=float(np.exp(q[5])), load_exponent=float(q[6]), load_reference=20.)
    else:
        friction = dict(mu0=float(coef[0]), a=float(coef[1]), b=float(coef[2]), viscous=float(coef[3]),
                        slip=float(np.exp(q[0])), theta0=float(np.exp(q[1])), vfloor=float(np.exp(q[2])))
    metadata = dict(family_known=True, stronger_structural_prior=True, reference_parameters_used=False,
                    fit_scope='closed loaded isolated contact, known preparation form; opening recovery unidentifiable',
                    target_label_inputs='N measured only as friction input; normal head fitted on B without N input',
                    family=world, dt=dt, vreg=VREG, seconds=time.perf_counter()-started,
                    max_seconds=max_seconds, restarts_requested=restarts, seed=seed,
                    normal_candidates=nc, friction_candidates=fc,
                    normal_validation_rmse=normal_fit['validation_rmse'],
                    friction_validation_rmse=friction_fit['validation_rmse'],
                    selection='training-only nonlinear optimization; whole isolated validation selects restart',
                    maxwell_load_gauge='20 N fixed; overall load scale is confounded with identified mu coefficients')
    return StateID(world, normal, friction, metadata)


def _local_validation_scores(model, validation, dt):
    """Use deployed clipped-force API to score candidate heads separately."""
    errors = {}
    for head in ('normal', 'friction'):
        x = validation[head]
        d = x.get('delta', np.full_like(x['target'], .02))
        rate = x.get('rate', np.zeros_like(d))
        velocity = x.get('velocity', np.zeros_like(d))
        measured_n = x['target'] if head == 'normal' else x['normal']
        state = model.initial(d[:, 0], velocity[:, 0], measured_n[:, 0])
        squared = 0.
        for t in range(d.shape[1]):
            n, ft, state = model.step(d[:, t], rate[:, t], velocity[:, t], state,
                                      normal_override=measured_n[:, t] if head == 'friction' else None, dt=dt)
            prediction = n if head == 'normal' else ft
            squared += float(np.sum((prediction-x['target'][:, t])**2))
        errors[head] = float(np.sqrt(squared/x['target'].size))
    return errors


def fit_blind_state_id(train, validation, dt=.0025, max_seconds=120., seed=0, restarts=3):
    """Library selection takes NO true-world argument; every head sees both families.

    Both families receive identical measured train/validation sequences. Linear
    coefficients and dynamic parameters use train only. Isolated validation
    independently selects the normal and friction family, with a fixed tie order.
    """
    started = time.perf_counter()
    families = ('maxwell', 'boucwen')
    models = {}; scores = {}
    for family in families:
        models[family] = fit_state_id(family, train, validation, dt=dt, max_seconds=max_seconds/2,
                                     seed=seed, restarts=restarts)
        scores[family] = _local_validation_scores(models[family], validation, dt)
    normal_family = min(families, key=lambda family: scores[family]['normal'])
    friction_family = min(families, key=lambda family: scores[family]['friction'])
    metadata = dict(selection='independent isolated-validation selection per head; fixed tie order maxwell,boucwen',
                    candidates=list(families), all_validation_rmse=scores, selected_normal_family=normal_family,
                    selected_friction_family=friction_family, true_family_label_used=False,
                    candidate_equations_known=True, stronger_structural_prior=True,
                    reference_parameters_used=False, max_seconds=max_seconds,
                    seconds=time.perf_counter()-started, seed=seed, restarts_per_family=restarts,
                    normal_validation_rmse=scores[normal_family]['normal'],
                    friction_validation_rmse=scores[friction_family]['friction'],
                    normal_candidates=models[normal_family].metadata['normal_candidates'],
                    friction_candidates=models[friction_family].metadata['friction_candidates'],
                    fit_scope='closed loaded contact; known preparation family for each candidate',
                    initialization='full causal prefix from declared preparation, no hidden material state sensor',
                    candidate_models={family: dict(normal=model.normal, friction=model.friction,
                                                  metadata=model.metadata) for family, model in models.items()})
    return CompositeStateID(models[normal_family], models[friction_family], metadata)
