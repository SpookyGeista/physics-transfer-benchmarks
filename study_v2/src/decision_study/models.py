"""Frozen local predictors using measured calibration only, with causal load history."""
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .common import FI, NI, VREG


def mlp(inp, width, out):
    return nn.Sequential(nn.Linear(inp, width), nn.Tanh(), nn.Linear(width, width),
                         nn.Tanh(), nn.Linear(width, out))


def replace_normal(f, normal):
    """Preserve past filtered N when replacing the current normal-force interface."""
    out = f.clone() if isinstance(f, torch.Tensor) else f.copy()
    change = normal / 20 - out[:, 9]
    out[:, 10:13] += change[:, None]
    out[:, 9] = normal / 20
    return out


class Neural(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind in ('modular', 'no_history'):
            self.normal_net = mlp(5, 24, 1)
            self.friction_net = mlp(8, 24, 1)
        elif kind == 'moe':
            self.experts = nn.ModuleList([mlp(13, 17, 12), mlp(13, 17, 12)])
            self.gate = mlp(13, 8, 2)
            self.head = nn.Linear(12, 2)
        elif kind == 'joint':
            self.net = mlp(13, 33, 2)
        else:
            raise ValueError(kind)

    def outputs(self, f):
        if self.kind == 'no_history':
            f = f.clone()
            f[:, 3:9] = 0
            f[:, 10:13] = 0
        if self.kind in ('modular', 'no_history'):
            return torch.cat((self.normal_net(f[:, NI]), self.friction_net(f[:, FI])), -1)
        if self.kind == 'moe':
            # Limited PRISM-WM-inspired control, not an original-paper reproduction.
            a, b = [expert(f) for expert in self.experts]
            a = nn.functional.normalize(a, dim=-1, eps=1e-6)
            b = nn.functional.normalize(b - (a*b).sum(-1, keepdim=True)*a, dim=-1, eps=1e-6)
            gate = torch.softmax(self.gate(f), -1)
            return self.head(gate[:, 0, None]*a + gate[:, 1, None]*b)
        return self.net(f)

    def normal(self, f):
        f = f.clone()
        f[:, 9:13] = 0  # current AND historical normal measurements unavailable to this head
        return torch.relu(10*f[:, 0] + 20*self.outputs(f)[:, 0]) * (f[:, 0] > 0)

    def friction(self, f):
        mu = nn.functional.softplus(self.outputs(f)[:, 1] - .87)
        return -20*torch.relu(f[:, 9])*mu*torch.tanh(.5*f[:, 2]/VREG)

    @torch.no_grad()
    def predict(self, f):
        f = torch.as_tensor(f, dtype=torch.float32)
        normal = self.normal(f)
        friction = self.friction(replace_normal(f, normal))
        return normal.cpu().numpy(), friction.cpu().numpy()

    @torch.no_grad()
    def predict_local(self, f):
        f = torch.as_tensor(f, dtype=torch.float32)
        return self.normal(f).cpu().numpy(), self.friction(f).cpu().numpy()


def normal_basis_v1(f):
    d = f[:, 0]
    return np.column_stack([d, f[:, 1], d**3, f[:, 3:6], np.tanh(f[:, 3:6]), d**2])


def friction_basis_v1(f):
    v = f[:, 2]*.5
    hv = v[:, None] - .5*f[:, 6:9]
    return np.column_stack([np.ones(len(f)), np.exp(-(v/.16)**2), 1/(1+(v/.12)**2),
                            np.abs(hv), np.tanh(f[:, 9]*20/15), np.abs(v),
                            v/(np.tanh(v/VREG)+np.where(v >= 0, 1e-5, -1e-5)) /
                            np.maximum(f[:, 9]*20, 1)])


def polynomial_basis(values, degree):
    """Complete monomials, including intercept, with deterministic feature ordering."""
    columns = [np.ones(len(values))]
    for power in range(1, degree+1):
        columns.extend(np.prod(values[:, ix], axis=1)
                       for ix in itertools.combinations_with_replacement(range(values.shape[1]), power))
    return np.column_stack(columns)


def flex_basis(f, head):
    if head == 'normal':
        return polynomial_basis(f[:, NI].astype(np.float64), 3)
    values = f[:, FI].astype(np.float64)
    return np.column_stack([polynomial_basis(values, 2), np.abs(values)])


class Classical:
    def __init__(self, kind='nominal', params=None):
        self.kind = kind
        self.params = params if params is not None else {'k': 500., 'c': 0., 'mu': .35}

    def _normal(self, f):
        if self.kind in ('nominal', 'identified'):
            n = self.params['k']*.02*f[:, 0] + self.params['c']*.1*f[:, 1]
        elif self.kind == 'analytic_v1':
            n = normal_basis_v1(f) @ np.asarray(self.params['normal'])
        else:
            fit = self.params['normal']
            n = 20*((flex_basis(f, 'normal')-fit['mean'])/fit['scale']) @ np.asarray(fit['coef'])
        return np.maximum(n, 0)*(f[:, 0] > 0)

    def _friction(self, f):
        if self.kind in ('nominal', 'identified'):
            mu = self.params['mu']
        elif self.kind == 'analytic_v1':
            mu = np.maximum(friction_basis_v1(f) @ np.asarray(self.params['friction']), 0)
        else:
            fit = self.params['friction']
            mu = np.maximum(((flex_basis(f, 'friction')-fit['mean'])/fit['scale']) @ np.asarray(fit['coef']), 0)
        return -np.maximum(f[:, 9]*20, 0)*mu*np.tanh(f[:, 2]*.5/VREG)

    def normal(self, f):
        if isinstance(f, torch.Tensor):
            return torch.as_tensor(self._normal(f.detach().cpu().numpy()), dtype=f.dtype, device=f.device)
        return self._normal(np.asarray(f))

    def friction(self, f):
        if isinstance(f, torch.Tensor):
            return torch.as_tensor(self._friction(f.detach().cpu().numpy()), dtype=f.dtype, device=f.device)
        return self._friction(np.asarray(f))

    def predict(self, f):
        f = np.asarray(f)
        normal = self._normal(f)
        return normal, self._friction(replace_normal(f, normal))

    def predict_local(self, f):
        f = np.asarray(f)
        return self._normal(f), self._friction(f)


def _fit_ridge_head(head, f, target, vf, vtarget, lambdas):
    basis = flex_basis(f, head)
    vbasis = flex_basis(vf, head)
    mean, scale = basis.mean(0), basis.std(0)
    mean[0], scale[0] = 0., 1.
    scale = np.where(scale < 1e-8, 1., scale)
    z, vz = (basis-mean)/scale, (vbasis-mean)/scale
    if head == 'normal':
        y = target/20
    else:
        # Fit measured tangential force directly; never divide noisy labels by v or N.
        z = z * (-np.maximum(f[:, 9]*20, 0)*np.tanh(f[:, 2]*.5/VREG)/10)[:, None]
        y = target/10
    gram, rhs = z.T @ z/len(z), z.T @ y/len(z)
    reg = np.eye(gram.shape[0]); reg[0, 0] = 0
    candidates = []
    for strength in lambdas:
        coef = np.linalg.solve(gram + strength*reg + 1e-12*np.eye(len(reg)), rhs)
        if head == 'normal':
            pred = np.maximum(20*(vz@coef), 0)*(vf[:, 0] > 0)
            loss = np.mean(((pred-vtarget)/20)**2)
        else:
            pred = -np.maximum(vf[:, 9]*20, 0)*np.maximum(vz@coef, 0)*np.tanh(vf[:, 2]*.5/VREG)
            loss = np.mean(((pred-vtarget)/10)**2)
        candidates.append((float(loss), float(strength), coef))
    loss, strength, coef = min(candidates, key=lambda item: item[0])
    return dict(mean=mean.tolist(), scale=scale.tolist(), coef=coef.tolist(),
                selected_lambda=strength, validation_loss=loss,
                candidates=[dict(lambda_=l, validation_loss=v) for v, l, _ in candidates])


def fit_classical(kind, f, n, ft, mask_n, mask_f, validation=None):
    from scipy.optimize import nnls
    if kind == 'nominal':
        return Classical()
    if kind == 'identified':
        coef = nnls(np.column_stack([f[mask_n, 0]*.02, f[mask_n, 1]*.1]), n[mask_n])[0]
        reg = -np.maximum(f[mask_f, 9]*20, 0)*np.tanh(f[mask_f, 2]*.5/VREG)
        mu = max(0., float(np.dot(reg, ft[mask_f])/max(np.dot(reg, reg), 1e-12)))
        return Classical(kind, dict(k=float(coef[0]), c=float(coef[1]), mu=mu))
    if kind == 'analytic_v1':
        cn = nnls(normal_basis_v1(f[mask_n]), n[mask_n])[0]
        use = mask_f & (np.abs(f[:, 2]) > .015) & (f[:, 9]*20 > 1)
        y = -ft[use]/(f[use, 9]*20*np.tanh(f[use, 2]*.5/VREG))
        cf = nnls(friction_basis_v1(f[use]), y)[0]
        return Classical(kind, dict(normal=cn.tolist(), friction=cf.tolist()))
    if kind != 'analytic_flex' or validation is None:
        raise ValueError('analytic_flex requires isolated validation')
    vf, vn, vft, vmn, vmf = validation
    lambdas = [1e-6, 1e-4, 1e-2, 1., 100.]
    normal = _fit_ridge_head('normal', f[mask_n], n[mask_n], vf[vmn], vn[vmn], lambdas)
    friction = _fit_ridge_head('friction', f[mask_f], ft[mask_f], vf[vmf], vft[vmf], lambdas)
    return Classical(kind, dict(normal=normal, friction=friction,
                               basis='normal: all monomials degree<=3 in NI; friction: degree<=2 in FI plus abs(FI)',
                               lambda_grid=lambdas))


def parameter_count(model):
    if isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters())
    if model.kind in ('nominal', 'identified'):
        return 3
    if model.kind == 'analytic_v1':
        return len(model.params['normal']) + len(model.params['friction'])
    return sum(len(model.params[head]['coef']) for head in ('normal', 'friction'))


def freeze(model):
    if isinstance(model, nn.Module):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def load_model(directory):
    directory = Path(directory)
    if (directory/'classical.json').exists():
        return Classical(**json.loads((directory/'classical.json').read_text()))
    state = torch.load(directory/'model.pt', map_location='cpu', weights_only=True)
    model = Neural(state['kind'])
    model.load_state_dict(state['state'])
    return freeze(model)
