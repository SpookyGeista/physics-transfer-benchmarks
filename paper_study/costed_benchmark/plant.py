"""Truth-only prepared native plant; never import from inference/model fitting.

Normal force is pre-update at current compression/rate. Tangential force is the
implicit displacement-work-equivalent interval mean. PI history commits along
the actual native displacement; solver/native/energy discrepancies stay visible.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "src"), str(ROOT / "alternatives"), str(ROOT.parent / "study_v2/src")]
from geometry import cam
from profiles import profile_commands
from implicit_stop_coupling import solve_velocity
from multistate_friction import MultiStateFriction, PARAMETER_SETS
from decision_study.reference import Reference


class CouplingFailure(RuntimeError):
    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


class PreparedPlant:
    def __init__(self, normal_world, parameter_set, factors, cfg):
        self.n, self.dt, self.mass = len(factors), cfg["dt"], cfg["mass_kg"]
        self.normal_world, self.parameter_set = normal_world, parameter_set
        self.normal = Reference(normal_world, self.n, cfg["initial_compression_m"])
        self.initial_normal = copy.deepcopy(self.normal).force(
            cfg["initial_compression_m"], 0., 0., friction_enabled=False, dt=self.dt)[0]
        seeds = {row["preparation_seed"] for row in factors}
        if len(seeds) != 1:
            raise ValueError("A plant batch requires one preparation seed")
        seed = seeds.pop()
        self.preparation = profile_commands(seed, [r["profile_index"] for r in factors],
            dt=self.dt, duration=cfg["preparation_seconds"],
            waypoint_limit=cfg["preparation_waypoint_limit_m"])
        self.friction = MultiStateFriction(parameter_set, self.n, initial_normal=self.initial_normal)
        for step in range(self.preparation["v"].shape[1]):
            self.friction.force(self.preparation["v"][:, step], self.initial_normal, self.dt)
        self.preparation_friction_audit = self.friction.audit()
        self.normal_updates = 0

    def observe_normal(self, delta, rate):
        return copy.deepcopy(self.normal).force(delta, rate, 0., friction_enabled=False, dt=self.dt)[0]

    def preparation_audit(self, initial_compression=.02):
        """Evaluator-only legacy-compatible command arrays with final boundary."""
        prep = self.preparation
        v = np.column_stack((prep["v"], np.zeros(self.n)))
        shape = v.shape
        return dict(normal_world=self.normal_world, friction_world=self.parameter_set,
            rows=prep["rows"], commands=dict(time=prep["time"], x=prep["x"], v=v,
            normal=np.broadcast_to(self.initial_normal[:, None], shape).copy(),
            delta=np.full(shape, initial_compression), rate=np.zeros(shape)),
            scope="Truth/evaluator only. Initial N is fresh normal force at delta=.02, rate=0. "
                  "The normal module remains freshly initialized after the separate friction fixture.")

    def step(self, rig, current, u, d, d_dot, next_d, amplitude, geometry, cfg):
        dt, mass = self.dt, self.mass
        g, slope = cam(current[:, 0], amplitude, geometry)
        delta, rate = d+g, d_dot+slope*current[:, 1]
        stop_before = self.friction.audit()
        # Exactly one real normal transition. Candidate solves never advance it.
        normal, _ = self.normal.force(delta, rate, 0., friction_enabled=False, dt=dt)
        self.normal_updates += 1
        capacity = normal * sum(PARAMETER_SETS[self.parameter_set].friction_capacity)
        def average(displacement):
            return self.friction.clone().force(displacement/dt, normal, dt)
        solved_v, applied_ft, solver_residual = solve_velocity(
            current[:, 1], u, slope, normal, capacity, average,
            dt=dt, mass=mass, iterations=cfg["solver_iterations"])
        force_external = u-slope*normal
        net = force_external+applied_ft
        record = dict(state=current.copy(), normal=normal, friction=applied_ft, u=u.copy(),
            d=np.full(self.n, d), d_dot=np.full(self.n, d_dot), delta=delta, compression_rate=rate,
            slope=slope, net_force=net, solver_velocity=solved_v, solver_residual_N=solver_residual,
            solver_iterations=np.full(self.n, cfg["solver_iterations"]),
            solver_bracket_width_m_s=2*dt*capacity/mass,
            friction_capacity=capacity, stop_state_before=stop_before["deflection"],
            stop_energy_before_J=stop_before["stored_energy"],
            normal_updates=np.full(self.n, self.normal_updates))
        if not np.isfinite(net).all() or np.max(abs(solver_residual)) > cfg["max_solver_residual_N"]:
            raise CouplingFailure("Implicit force solution failed predeclared residual/finite check", record)
        next_state = rig.advance(net)
        record["next_state"] = next_state
        if not np.isfinite(next_state).all():
            raise CouplingFailure("Nonfinite native state; PI history not committed with NaN", record)
        dx, vn = next_state[:, 0]-current[:, 0], next_state[:, 1]
        committed_ft = self.friction.force(dx/dt, normal, dt)
        stop_after = self.friction.audit()
        step = stop_after["last_step"]
        new_g, _ = cam(next_state[:, 0], amplitude, geometry)
        delta_energy = stop_after["stored_energy"]-stop_before["stored_energy"]
        kinetic_change = .5*mass*(vn**2-current[:, 1]**2)
        numerical_dissipation = .5*mass*(vn-current[:, 1])**2
        momentum_residual = mass*(vn-current[:, 1])/dt-net
        kinematic_residual = dx-dt*vn
        energy_residual = force_external*dx-kinetic_change-delta_energy-step["dissipation"]-numerical_dissipation
        decomposed = (committed_ft-applied_ft)*dx+net*kinematic_residual-dt*momentum_residual*vn
        record.update(actual_displacement=dx, friction_commit=committed_ft,
            friction_commit_minus_applied=committed_ft-applied_ft,
            native_velocity_minus_solver=vn-solved_v,
            native_position_residual_m=kinematic_residual, native_momentum_residual_N=momentum_residual,
            stop_state_after=stop_after["deflection"], stop_energy_after_J=stop_after["stored_energy"],
            stop_work_input_J=step["work_input"], stop_dissipation_J=step["dissipation"],
            stop_load_release_J=step["load_release"], stop_slip_dissipation_J=step["slip_dissipation"],
            stop_material_residual_J=step["energy_residual"],
            external_work_J=force_external*dx, kinetic_change_J=kinetic_change,
            numerical_dissipation_J=numerical_dissipation, tangential_balance_residual_J=energy_residual,
            balance_residual_decomposition_J=decomposed,
            energy_decomposition_error_J=energy_residual-decomposed,
            normal_drive_work_zoh_J=normal*d_dot*dt,
            normal_drive_work_discrete_J=normal*(next_d-d),
            normal_geometry_work_J=normal*(new_g-g),
            normal_slope_work_J=normal*slope*dx,
            normal_geometry_discretization_J=normal*(new_g-g-slope*dx),
            normal_material_work_discrete_J=normal*(next_d-d+new_g-g),
            actuator_work_J=u*dx, normal_reaction_work_J=-normal*slope*dx,
            friction_applied_work_J=applied_ft*dx)
        if not all(np.isfinite(value).all() for value in record.values()):
            raise CouplingFailure("Nonfinite committed material/energy quantity", record)
        return record
