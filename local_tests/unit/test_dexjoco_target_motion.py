"""Agent as policy's target-level DexJoCo motion (dexjoco_move_ee, dexjoco_hand): the rows the
server generates, its settle loop, and which interface each contract exposes."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

pytest.importorskip('scipy')
from scipy.spatial.transform import Rotation  # noqa: E402

from services.dexjoco import tools as T  # noqa: E402
from services.dexjoco.backend import DexjocoBackend  # noqa: E402
from services.mcp_contract import Contract  # noqa: E402


class FakeSim:
    """Single-arm state: the flange closes half the gap to its row target each step; joints listed in
    `blocked` stop at 0.3 rad (an object in the way)."""
    def __init__(self, lag=0.5, blocked=()):
        self.episode_id, self.step_id, self.active = 'e1', 0, True
        rotation = Rotation.from_euler('x', 180, degrees=True)
        self.position, self.quat = np.array([0.0, 0.0, 1.2]), rotation.as_quat(scalar_first=True)
        self.hand = np.zeros(16)
        self.lag, self.blocked, self.rows = lag, set(blocked), []

    def state(self):
        return [*self.position, *self.quat, *self.hand]

    def request(self, op, **data):
        if op == 'observe':
            return {'state': self.state()}
        assert op == 'chunk_step'
        steps = []
        for row in data['actions']:
            self.rows.append(np.array(row))
            self.position = self.position + self.lag * (row[:3] - self.position)
            self.quat = Rotation.from_rotvec(row[3:6]).as_quat(scalar_first=True)
            target = np.array(row[6:22])
            for j in range(16):
                self.hand[j] = min(target[j], 0.3) if j in self.blocked else target[j]
            self.step_id += 1
            steps.append({'step_id': self.step_id, 'executed_action': list(row), 'terminated': False})
        return {'frames': [{'state': self.state(), 'step_id': self.step_id}], 'steps': steps,
                'frame_count': 1, 'terminated': False, 'truncated': False}


def backend(sim):
    b = object.__new__(DexjocoBackend)
    b.info, b.sim, b._raw, b._command = {'dual_arm': False}, sim, None, None
    b.contract = SimpleNamespace(action_interface='targets')
    return b


def test_move_ee_interpolates_at_the_speed_and_settles_within_tolerance():
    sim = FakeSim()
    result, summary = backend(sim)._move_ee({'delta_m': [0.1, 0.0, -0.1], 'speed': 'precision',
                                              'position_tolerance_m': 0.005})
    step = np.linalg.norm(np.diff([r[:3] for r in sim.rows[:5]], axis=0), axis=1)
    assert np.all(step <= T.SPEEDS['precision'][0] + 1e-9)
    assert summary['reached'] and summary['position_error_m'] <= 0.005
    assert summary['interpolated_steps'] == int(np.ceil(np.hypot(0.1, 0.1) / T.SPEEDS['precision'][0]))
    assert summary['steps_used'] == len(sim.rows) == summary['interpolated_steps'] + summary['settle_steps']
    np.testing.assert_allclose(sim.rows[-1][:3], [0.1, 0.0, 1.1])


def test_moves_keep_a_closed_hand_closed_and_report_blocked_joints():
    sim = FakeSim(blocked={0, 1})
    b = backend(sim)
    _, hand = b._hand({'joints_rad': [1.0] * 16, 'steps': 10, 'settle_steps': 2})
    assert hand['blocked_joints'] == ['ff0', 'ff1'] and hand['steps_used'] == 12
    b._move_ee({'delta_m': [0.0, 0.0, 0.05]})
    assert all(np.allclose(r[6:22], 1.0) for r in sim.rows[12:])  # targets stay closed, not the measured 0.3
    b._run_rows(np.repeat(b._base_row()[None], 3, axis=0))
    assert np.allclose(sim.rows[-1][6:22], 1.0)


def test_a_long_slow_move_is_rejected_before_any_step():
    sim = FakeSim()
    with pytest.raises(Exception, match='needs'):
        backend(sim)._move_ee({'position_m': [2.0, 0.0, 1.2], 'speed': 'precision'})
    assert not sim.rows


def test_rotation_targets_are_reached_without_rotvec_jumps():
    sim = FakeSim()
    target = Rotation.from_euler('xz', [180, 90], degrees=True)
    _, summary = backend(sim)._move_ee({'rotation': {'quaternion_wxyz': target.as_quat(scalar_first=True).tolist()}})
    rotvecs = np.array([r[3:6] for r in sim.rows])
    assert np.max(np.linalg.norm(np.diff(rotvecs, axis=0), axis=1)) < 0.2
    assert summary['rotation_error_rad'] < 0.05


@pytest.mark.parametrize('policy,present,absent', [
    (False, {'dexjoco_step', 'dexjoco_preview_run'}, {'dexjoco_move_ee', 'dexjoco_hand'}),
    (True, {'dexjoco_move_ee', 'dexjoco_hand', 'dexjoco_hold'}, {'dexjoco_step', 'dexjoco_preview_run'})])
def test_each_mode_exposes_its_own_action_interface(policy, present, absent):
    config = SimpleNamespace(observation_profile='rgbd', benchmark='dexjoco', agent_formal=policy,
                             agent_policy=policy, external_perception=False)
    names = Contract.from_config(config, isolated=True).robot_names
    assert present <= names and not (absent & names)


class SaggingSim(FakeSim):
    """The flange settles 1.5 cm below its commanded height, like the official controller under load."""
    def request(self, op, **data):
        reply = super().request(op, **data)
        if op == 'chunk_step':
            self.position[2] = data['actions'][-1][2] - 0.015
            reply['frames'][-1]['state'] = self.state()
        return reply


def test_settling_leads_a_lagging_controller_and_bounds_the_lead():
    sim = SaggingSim()  # a steady 1.5 cm sag: the lead (gain 1.5) brings it under 1 cm
    _, summary = backend(sim)._move_ee({'delta_m': [0.0, 0.0, -0.1]})
    assert summary['reached'] and summary['position_error_m'] <= 0.01
    sim = FloorSim()
    _, summary = backend(sim)._move_ee({'delta_m': [0.0, 0.0, -0.2]})
    assert not summary['reached'] and summary['position_error_m'] > 0.05
    assert np.linalg.norm(summary['settle_lead_m']) <= T.MAX_LEAD_M + 1e-9


class FloorSim(FakeSim):
    """An obstacle: the flange cannot go below z = 1.1."""
    def request(self, op, **data):
        reply = super().request(op, **data)
        if op == 'chunk_step':
            self.position[2] = max(self.position[2], 1.1)
            reply['frames'][-1]['state'] = self.state()
        return reply


def test_a_move_blocked_by_contact_stops_settling_early():
    sim = FloorSim()
    _, summary = backend(sim)._move_ee({'delta_m': [0.0, 0.0, -0.2], 'max_settle_steps': 120})
    assert summary['stalled'] and not summary['reached'] and summary['settle_steps'] <= 30


class SlowSim(FakeSim):
    """A first-order lag with the official controller's time constant (about 20 steps)."""
    def __init__(self):
        super().__init__(lag=1 / 20)


def test_the_lead_speeds_up_a_slow_controller():
    _, summary = backend(SlowSim())._move_ee({'delta_m': [0.0, 0.0, -0.1]})
    assert summary['reached'] and summary['settle_steps'] <= 30
