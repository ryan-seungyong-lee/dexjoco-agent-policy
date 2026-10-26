"""DexJoCo backend: catalog, official layouts, validation, geometry, privacy and host separation.

No simulator: the backend is exercised against a fake simulator process. Catalog tests read the
official evaluation configs from the pinned DexJoCo checkout (scripts/bootstrap_sources.sh).
"""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from services.benchmarks import BENCHMARKS, DEXJOCO_TOOLS
from services.dexjoco import tasks as TK
from services.dexjoco import tools as T
from services.mcp_contract import Contract
from services.simulator.io import RejectedBeforeAction

ROOT = Path(__file__).resolve().parents[2]
needs_source = pytest.mark.skipif(not (TK.source_root() / "configs").is_dir(),
                                  reason="pinned DexJoCo checkout not installed (scripts/bootstrap_sources.sh)")


# ------------------------------------------------------------------ contract
ROW_TOOLS = DEXJOCO_TOOLS - BENCHMARKS['dexjoco'].target_tools  # The default 'rows' interface.


def test_contract_selects_dexjoco_tools():
    contract = Contract(benchmark='dexjoco')
    assert contract.robot_names == ROW_TOOLS
    assert Contract(observation_profile='rgb-only', benchmark='dexjoco').robot_names == \
        ROW_TOOLS - {'dexjoco_pixel_to_position'}
    with pytest.raises(PermissionError):
        contract.require_robot('not_a_tool', {})
    assert contract.scoped_tools >= {'dexjoco_step', 'dexjoco_hold', 'start_episode'}
    assert Contract(benchmark='dexjoco').manifest()['benchmark'] == 'dexjoco'


def test_registry_entry_matches_the_tool_module():
    bench = BENCHMARKS['dexjoco']
    assert T.ROBOT_TOOLS == bench.robot_tools and T.MUTATING == bench.mutating
    assert bench.robot['benchmark'] == bench.label and not bench.previewable and bench.robot['plan'] == ''
    assert bench.graded_score is False and bench.partial_credit is False
    assert {t['name'] for t in bench.definitions()} == DEXJOCO_TOOLS
    assert set(T.CAMERAS) == set(bench.robot['cameras'])


# ------------------------------------------------------------------- catalog
def test_official_layouts_are_the_evaluation_episodes():
    assert TK.FORMAL_LAYOUTS == 150
    assert TK.layout('pick_bucket', 1, 0) == {'seed': 0, 'episode': 0}
    assert TK.layout('pick_bucket', 1, 49) == {'seed': 0, 'episode': 49}
    assert TK.layout('pick_bucket', 1, 50) == {'seed': 1, 'episode': 0}
    assert TK.layout('pick_bucket_random', 1, 149) == {'seed': 2, 'episode': 49}
    with pytest.raises(ValueError, match='outside'):
        TK.layout('pick_bucket', 1, 150)
    exploration = {TK.layout('pick_bucket', 0, i)['seed'] for i in range(1000)}
    assert len(exploration) == 1000 and not exploration & set(TK.OFFICIAL_SEEDS)
    assert all(TK.layout('pick_bucket', 0, i)['episode'] == 0 for i in (0, 999))


def test_catalog_limits_and_variants():
    names = TK.task_names()
    assert len(names) == 22 and 'pick_bucket' in names and 'bimanual_hanoi_random' in names
    assert TK.official_task('pick_bucket_random') == ('pick_bucket', 'rand_full')
    assert TK.native_step_limit('pick_bucket') == 1200
    assert TK.native_step_limit('click_mouse') == 1170  # the official 30-step pre-roll counts toward the cap
    assert TK.action_dim('bimanual_hanoi') == 44 and TK.state_dim('pick_bucket') == 23
    timeouts = TK.task_timeouts('pick_bucket')
    assert timeouts['native_step_limit'] == 1200 and timeouts['episode_seconds'] > timeouts['startup_seconds']
    # Task names never collide with another benchmark's catalog.
    others = {t for name, b in BENCHMARKS.items() if name != 'dexjoco' for t in b.task_names()}
    assert not others & set(names)


@needs_source
def test_official_configs_and_task_context():
    from services.dexjoco.task_rules import task_context
    config = TK.official_config('pick_bucket_random')
    assert config['camera_mapping'] == {'base': 'random_camera', 'wrist': 'wrist'}
    public = task_context('pick_bucket', task_rules='public')
    assert f"**Instruction:** {TK.official_config('pick_bucket')['prompt']}" in public
    assert 'There is no partial credit' in public and 'Task rules' not in public
    assert '150 official' in public
    summary = task_context('pick_bucket_random', task_rules='code-summary', live_formal=True)
    assert 'third-person camera' in summary and 'Wall-clock' not in summary and 'no reset' in summary
    for task in TK.task_names():
        assert task_context(task, task_rules='code-summary').count('**Scene variation:**') == 1


@needs_source
def test_task_robot_words_follow_the_official_cameras():
    bench = BENCHMARKS['dexjoco']
    assert bench.robot_facts('bimanual_hanoi')['cameras'] == ('ego', 'wrist_left', 'wrist_right')
    assert bench.robot_facts('bimanual_hanoi')['arms'] == 'dual'
    assert bench.robot_facts('click_mouse')['overview_camera'] == 'ego_right'
    assert bench.robot_facts('pick_bucket_random')['wrist_images'] == '`wrist`'


# ---------------------------------------------------------------- validation
def test_request_validation_rejects_malformed_calls_before_dispatch():
    row = [0.0] * 22
    T.validate_request(T.STEP, {'actions': [row] * 3}, action_dim=22)
    for bad in ({'actions': []}, {'actions': [[0.0] * 21]}, {'actions': [row, [0.0] * 44]},
                {'actions': [[float('nan')] * 22]}, {'actions': [row] * (T.MAX_ACTION_ROWS + 1)}, {'rows': [row]}):
        with pytest.raises(RejectedBeforeAction):
            T.validate_request(T.STEP, bad, action_dim=22)
    with pytest.raises(RejectedBeforeAction, match='44-D'):
        T.validate_request(T.STEP, {'actions': [row]}, action_dim=44)
    for bad in ({'steps': 0}, {'steps': 1.5}, {'steps': T.MAX_ACTION_ROWS + 1}):
        with pytest.raises(RejectedBeforeAction):
            T.validate_request(T.HOLD, bad)
    with pytest.raises(RejectedBeforeAction):
        T.validate_request(T.PIXEL_TO_POSITION, {'observation_step_id': 0,
                                                 'selections': [{'camera': 'no_such_camera', 'u': 1, 'v': 1}]})


def test_pixel_and_world_projection_round_trip():
    intrinsics = {'matrix': [[500.0, 0, 320], [0, 500.0, 320], [0, 0, 1]]}
    # A camera at z = 2 looking straight down: OpenGL cameras look along their own -Z, so identity.
    pose = {'position_m': [0.1, -0.2, 2.0], 'quaternion_wxyz': [1.0, 0.0, 0.0, 0.0]}
    point = np.array([0.25, -0.1, 0.9])
    u, v, z = T.world_to_pixel(point, intrinsics, pose)
    assert z == pytest.approx(1.1)
    depth = np.full((640, 640), z, np.float32)
    back = T.pixel_to_world(depth, intrinsics, pose, u, v, 0)
    assert back['valid'] and np.allclose(back['position_m'], point, atol=1e-6)
    # Row 0 is the top of the image: a point further along the camera's +Y (up) has a smaller v.
    assert T.world_to_pixel(point + [0, 0.05, 0], intrinsics, pose)[1] < v
    assert T.world_to_pixel(point + [0.05, 0, 0], intrinsics, pose)[0] > u


@pytest.mark.parametrize('dual', [False, True])
def test_held_steps_report_the_row_stay_sends(dual):
    """session.hold_row is DexJoCoOpenPIEnv.stay()'s action for a held state, in its own words."""
    from scipy.spatial.transform import Rotation as R
    from services.dexjoco.session import hold_row
    state = np.random.default_rng(3).normal(size=46 if dual else 23)
    state[3:7] = [0.02, -0.99, 0.05, 0.01]  # Near the start pose: a rotation of about pi.
    def rotvec(quat):
        return R.from_quat(quat, scalar_first=True).as_rotvec()
    if dual:
        expected = np.concatenate([state[0:3], rotvec(state[3:7]), state[14:30],
                                   state[7:10], rotvec(state[10:14]), state[30:46]])
    else:
        expected = np.concatenate([state[0:3], rotvec(state[3:7]), state[7:23]])
    row = hold_row(state, dual)
    assert isinstance(row, list) and len(row) == T.action_layout(dual)['dim']
    np.testing.assert_allclose(row, expected, atol=1e-12)


# ------------------------------------------------------------------- backend
class FakeProcess:
    """In-memory stand-in for services.dexjoco.process.DexjocoProcess."""

    def __init__(self, output_root, success=False, cameras=('front', 'wrist')):
        self.output_root = Path(output_root)
        self.process = self.rpc = None
        self.episode_id, self.step_id, self.run_id, self.output_dir = None, 0, None, None
        self.finished, self.startup_seconds, self.success = False, 1.0, success
        self.metadata = {'cameras': list(cameras)}
        self.requests = []

    @property
    def active(self):
        return self.process is not None and not self.finished

    def start(self, *, task, collection, layout_id, run_id, **kwargs):
        self.task, self.collection, self.layout_id, self.run_id = task, collection, layout_id, run_id
        self.output_dir = self.output_root / run_id
        (self.output_dir / 'sim').mkdir(parents=True)
        self.process, self.episode_id, self.step_id, self.finished = object(), f'ep-{layout_id}', 0, False
        self._state()
        return {'episode_id': self.episode_id, 'step_id': 0, 'native_step_limit': 1200}

    def _state(self):
        (self.output_dir / 'sim' / 'operator_state.json').write_text(json.dumps({
            'episode_id': self.episode_id, 'step_id': self.step_id,
            'reward': {'native_success': self.success, 'native_score_percent': 100.0 if self.success else 0.0}}))

    def observation(self):
        cameras = self.metadata['cameras']
        obs = {c: np.full((8, 8, 3), 100, np.uint8) for c in cameras}
        obs.update({f'{c}_depth_m': np.ones((8, 8), np.float32) for c in cameras})
        obs.update(state=list(np.linspace(0, 1, 23)), instruction='Do the task',
                   camera_intrinsics={c: {'matrix': [[8, 0, 4], [0, 8, 4], [0, 0, 1]]} for c in cameras},
                   camera_poses={c: {'position_m': [0, 0, 2], 'quaternion_wxyz': [0, 1, 0, 0]} for c in cameras},
                   step_id=self.step_id)
        return obs

    def request(self, op, **kwargs):
        self.requests.append((op, kwargs))
        if op == 'observe':
            return self.observation()
        if op in ('chunk_step', 'hold'):
            count = len(kwargs['actions']) if op == 'chunk_step' else kwargs['steps']
            frames, steps = [], []
            for _ in range(count):
                self.step_id += 1
                steps.append({'step_id': self.step_id, 'executed_action': None, 'terminated': self.success,
                              'truncated': False, **({'success': True} if self.success else {})})
                if self.step_id % T.FRAME_STRIDE == 0:
                    frames.append({k: v for k, v in self.observation().items() if not k.endswith('_depth_m')})
                if self.success:
                    break
            frames.append(self.observation())
            self._state()
            return {'episode_id': self.episode_id, 'step_id': self.step_id, 'frames': frames, 'steps': steps,
                    'terminated': self.success, 'truncated': False, **({'success': True} if self.success else {})}
        raise AssertionError(op)

    def finish(self, reason):
        self.finished = True

    def stop(self):
        self.process = None
        self.finished = True


def backend_for(tmp_path, profile='rgbd', success=False, task='pick_bucket'):
    from services.dexjoco.backend import DexjocoBackend
    config = SimpleNamespace(root=tmp_path, task=task, benchmark='dexjoco', observation_profile=profile,
                             exploration_envs=1, subagents=0, eval_seed=0, formal_collection=1, sim_gpu='GPU-0',
                             formal=SimpleNamespace(wall_seconds=3000))
    backend = DexjocoBackend(config)
    backend.sim = FakeProcess(tmp_path / 'native' / 'results', success=success)
    return backend


PRIVATE_WORDS = ('native_score', 'native_success', 'official_seed', 'layout_id', 'privileged', 'env_info',
                 'initial_state', '"seed"')


def assert_public(value):
    text = json.dumps(value)
    for word in PRIVATE_WORDS:
        assert word not in text, word


def test_backend_start_step_hold_and_evaluate(tmp_path):
    backend = backend_for(tmp_path)
    value, images = backend.start(formal=True, seed=57)
    assert backend.sim.collection == 1 and backend.sim.layout_id == 57
    assert value['environment_spec']['episode_step_limit'] == 1200 and value['instruction'] == 'Do the task'
    assert len(images) == 4  # two RGB + two depth attachments
    assert_public(value)
    value, images = backend.call('dexjoco_step', {'actions': [[0.0] * 22] * 13})
    assert value['step_id'] == 13 and value['frame_sequence']['frame_count'] == 2 and len(images) == 4
    assert backend.sim.requests[-1][0] == 'chunk_step' and backend.sim.requests[-1][1]['actions'].shape == (13, 22)
    assert_public(value)
    manifest = json.loads((backend.frame_workspace / value['frame_sequence']['manifest_path']).read_text())
    from services.controller.recording import IMAGE_NAME
    names = [Path(item['path']).name for f in manifest['frames'] for item in f['files']]
    assert all(IMAGE_NAME.fullmatch(name) for name in names) and 'frame_000001_wrist_depth.png' in names
    value, _ = backend.call('dexjoco_hold', {'steps': 5})
    assert value['step_id'] == 18 and backend.sim.requests[-1] == ('hold', {'steps': 5})
    evaluation = backend.evaluate()
    assert evaluation['task_complete'] is False and evaluation['evaluation_kind'] == 'binary_task_completion'
    assert_public(evaluation)
    point = backend.call('dexjoco_pixel_to_position',
                         {'observation_step_id': 18, 'selections': [{'camera': 'front', 'u': 4, 'v': 4}]})[0]
    assert point['selections'][0]['valid'] and point['selections'][0]['frame'] == 'world'
    with pytest.raises(RejectedBeforeAction, match='Stale'):
        backend.call('dexjoco_pixel_to_position',
                     {'observation_step_id': 12, 'selections': [{'camera': 'front', 'u': 4, 'v': 4}]})
    with pytest.raises(RejectedBeforeAction, match='22-D'):
        backend.call('dexjoco_step', {'actions': [[0.0] * 44]})


def test_backend_rgb_only_removes_depth_and_calibration(tmp_path):
    backend = backend_for(tmp_path, 'rgb-only')
    value, images = backend.start(formal=False, seed=0)
    assert len(images) == 2 and backend.sim.collection == 0
    text = json.dumps(value)
    assert 'camera_intrinsics' not in text and 'camera_poses' not in text and '"depth"' not in text
    assert 'dexjoco_pixel_to_position' not in {t['name'] for t in backend.tools()}


def test_backend_reports_success_and_episode_end(tmp_path):
    backend = backend_for(tmp_path, success=True)
    backend.start(formal=False, seed=0)
    value, _ = backend.call('dexjoco_step', {'actions': [[0.0] * 22] * 4})
    assert value['transition']['episode_ended'] is True and value['transition']['task_complete'] is True
    assert value['step_id'] == 1  # the episode ended on the first row
    assert backend.evaluate()['task_complete'] is True


@needs_source
def test_status_reports_the_official_layouts(tmp_path):
    backend = backend_for(tmp_path, task='bimanual_hanoi')
    status, _ = backend.call('dexjoco_status', {})
    assert status['action_layout']['dim'] == 44 and status['state_layout']['dim'] == 46
    assert status['action_layout']['fields'][5]['joints'][:4] == ['rf0', 'rf1', 'rf2', 'rf3']
    assert status['instruction'] == TK.official_config('bimanual_hanoi')['prompt']


def test_gateway_lists_and_routes_dexjoco_tools(tmp_path):
    from services.controller.gateway import Gateway
    backend = backend_for(tmp_path)
    gateway = Gateway(backend, None, contract=backend.contract, audit=lambda *_: None, charge=lambda *_: None,
                      refund=lambda *_: None, budget=lambda: {})
    token = gateway.acquire()
    names = {t['name'] for t in gateway.handle(token, {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'})['result']['tools']}
    assert ROW_TOOLS <= names
    backend.start(formal=False, seed=0)
    reply = gateway.handle(token, {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call',
                                   'params': {'name': 'dexjoco_step', 'arguments': {'actions': [[0.0] * 22] * 2}}})
    assert json.loads(reply['result']['content'][0]['text'])['step_id'] == 2
    refused = gateway.handle(token, {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call',
                                     'params': {'name': 'not_a_tool', 'arguments': {'actions': []}}})
    assert 'error' in refused


def test_submit_description_follows_binary_scoring():
    from services.controller.frontend import BINARY_FORMAL, DEFINITIONS, GRADED_FORMAL, for_config
    config = SimpleNamespace(benchmark='dexjoco', agent_cli='codex', agent_formal=False)
    submit = next(t for t in for_config(DEFINITIONS, config) if t['name'] == 'submit')
    assert BINARY_FORMAL in submit['description'] and GRADED_FORMAL not in submit['description']


def test_settings_bound_the_official_layouts():
    from services.dexjoco.benchmark import validate_settings
    validate_settings({'formal_seed': 0, 'formal_episodes': 150})
    with pytest.raises(ValueError, match='150 official'):
        validate_settings({'formal_seed': 1, 'formal_episodes': 150})


# -------------------------------------------------------------- separation
def test_host_modules_never_import_the_simulator_stack():
    """Every host-side module imports with dexjoco, mujoco, gymnasium and openpi unavailable."""
    code = r'''
import importlib, sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'dexjoco', 'dexjoco_openpi_client', 'mujoco', 'gymnasium', 'openpi_client', 'glfw'}:
            raise ImportError('blocked ' + name)
sys.meta_path.insert(0, Block())
for module in ('services.benchmarks', 'services.mcp_contract', 'services.dexjoco.benchmark', 'services.dexjoco.tools',
               'services.dexjoco.tasks', 'services.dexjoco.process', 'services.dexjoco.backend',
               'services.dexjoco.task_rules', 'services.dexjoco.demonstrations', 'services.dexjoco.preview',
               'services.dexjoco.preview_session', 'services.dexjoco.rotation_math',
               'services.controller.backend', 'services.controller.gateway', 'services.controller.frontend',
               'services.controller.supervisor', 'services.exploration.research', 'services.workspace.build'):
    importlib.import_module(module)
print('ok')
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.strip() == 'ok', result.stderr


def test_dashboard_lists_dexjoco_tasks_and_cameras():
    from services.dashboard.sessions import CAMERAS, benchmark_of, timeouts
    assert benchmark_of('pick_bucket') == 'dexjoco'
    assert timeouts('pick_bucket')['native_step_limit'] == 1200
    assert {'front', 'random_camera', 'ego', 'wrist_left'} <= set(CAMERAS)
    from services.mcp_contract import GEOMETRY_KEYS
    assert {'front_depth_m', 'wrist_depth_m', 'ego_depth_m'} <= GEOMETRY_KEYS
