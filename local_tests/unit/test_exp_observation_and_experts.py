"""The experiment's added observation profiles (rgb-depth, rgb-camera) and expert contents
(video, human-video, human-full): what reaches the agent and what its workspace says."""
import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from services.mcp_contract import CAMERA_KEYS, DEPTH_KEYS, PROFILES, Contract, ObservationProfile  # noqa: E402

PACKET = {
    'state': [0.0] * 23, 'images': ['front'],
    'attachments': [{'kind': 'rgb', 'camera': 'front', 'content_index': 1},
                    {'kind': 'depth', 'camera': 'front', 'content_index': 2}],
    'depth': {'front': {'min_m': 0.4}}, 'front_depth_m': [[1.0]],
    'camera_intrinsics': {'front': {'matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}},
    'camera_poses': {'front': {'position_m': [0, 0, 1], 'quaternion_wxyz': [1, 0, 0, 0]}},
}
EXPECTED = {'rgbd': (True, True), 'rgb-depth': (True, False), 'rgb-camera': (False, True), 'rgb-only': (False, False)}


def test_profiles_are_the_four_arms():
    assert set(PROFILES) == set(EXPECTED)


@pytest.mark.parametrize('name', PROFILES)
def test_clean_keeps_exactly_the_profiles_inputs(name):
    depth, camera = EXPECTED[name]
    profile = ObservationProfile(name)
    assert (profile.depth, profile.camera_params, profile.geometry) == (depth, camera, depth and camera)
    cleaned = profile.clean(json.loads(json.dumps(PACKET)))
    assert ('depth' in cleaned) == depth and ('front_depth_m' in cleaned) == depth
    assert any(a['kind'] == 'depth' for a in cleaned['attachments']) == depth
    assert ('camera_intrinsics' in cleaned) == camera and ('camera_poses' in cleaned) == camera
    assert cleaned['state'] == PACKET['state'] and cleaned['images'] == ['front']
    assert not (set(map(str.lower, cleaned)) & (frozenset() if depth else DEPTH_KEYS))
    assert not (set(map(str.lower, cleaned)) & (frozenset() if camera else CAMERA_KEYS))


@pytest.mark.parametrize('name', PROFILES)
def test_pixel_tool_needs_depth_and_calibration(name):
    contract = Contract(observation_profile=name, benchmark='dexjoco')
    assert ('dexjoco_pixel_to_position' in contract.robot_names) == (name == 'rgbd')
    observe = next(t for t in contract.robot_definitions() if t['name'] == 'dexjoco_observe')
    depth, camera = EXPECTED[name]
    if not depth:
        assert 'depth attachments' not in observe['description']
    if not camera:
        assert 'calibration,' not in observe['description'] or 'no depth or calibration' in observe['description'] \
            or 'no calibration' in observe['description']


def _render(tmp_path, **arguments):
    from scripts.render_workspaces import auto_research
    target = tmp_path / 'workspace'
    auto_research(target, benchmark='dexjoco', claude=True, **arguments)
    return target


def _skill(workspace, name):
    path = workspace / '.agents' / 'skills' / name / 'SKILL.md'
    return path.read_text() if path.exists() else None


@pytest.mark.parametrize('profile,expected', [
    ('rgb-depth', 'RGB + depth (no calibration)'), ('rgb-camera', 'RGB + camera calibration (no depth)')])
def test_new_profiles_get_their_own_localization_skill(tmp_path, profile, expected):
    workspace = _render(tmp_path, profile=profile)
    assert expected in _skill(workspace, 'object-localization')
    assert 'dexjoco_pixel_to_position' not in _skill(workspace, 'object-localization')


@pytest.mark.parametrize('content', ['video', 'human-video', 'human-full'])
def test_expert_content_variants_render_and_validate(tmp_path, content):
    from scripts.render_workspaces import PROJECT as RP
    from services.workspace.validate import validate
    workspace = _render(tmp_path, experts=3, expert_content=content)
    human = content.startswith('human-')
    experts, human_skill = _skill(workspace, 'learn-from-experts'), _skill(workspace, 'learn-from-human-demos')
    assert (experts is None) == human and (human_skill is None) == (not human)
    if content == 'video':
        assert 'trajectory.npz' not in experts and 'videos/*.mp4' in experts
    if human:
        assert ('hand_trajectory.npz' in human_skill) == (content == 'human-full')
    known = frozenset(p.parent.name for p in (RP / 'agent_skills').rglob('SKILL.md'))
    tools = Contract(benchmark='dexjoco').tool_names
    problems = [p for p in validate(workspace, tools=tools, cli='claude', benchmark='dexjoco',
                                    expert_demonstrations=3, expert_content=content, known_skills=known)
                if 'mentions' not in p]  # tool-mention rules depend on the full case facts
    assert problems == []


def test_video_view_drops_actions_and_states(tmp_path):
    from services.dexjoco import demonstrations as D
    demos = []
    for name in ('a', 'b', 'c'):
        demo = tmp_path / 'experts' / name
        (demo / 'videos').mkdir(parents=True)
        (demo / 'videos' / 'base.mp4').write_bytes(b'v')
        (demo / 'frames').mkdir()
        (demo / 'frames' / 'index.json').write_text('{}')
        (demo / 'episode.json').write_text('{}')
        (demo / 'trajectory.npz').write_bytes(b'x')
        (demo / 'replay.zarr').mkdir()
        demos.append(demo)
    view = D._video_view(demos, tmp_path / 'experts-video', D.ROBOT_VIDEO_PARTS)
    for name in ('a', 'b', 'c'):
        assert sorted(p.name for p in (view / name).iterdir()) == ['episode.json', 'frames', 'videos']


def test_unattended_note_only_when_requested(monkeypatch):
    from harness.codex_cli.auto_research import unattended
    monkeypatch.delenv('RAX_UNATTENDED', raising=False)
    assert unattended('Go.') == 'Go.'
    monkeypatch.setenv('RAX_UNATTENDED', '1')
    assert 'submit' in unattended('Go.') and unattended('Go.').startswith('Go. No human operator')
    assert 'submit' not in unattended('Go.', agent_formal=True)


@pytest.mark.parametrize('experts', [0, 3])
def test_agent_policy_copy_has_no_exploring_past(tmp_path, experts):
    from scripts.render_workspaces import formal_copy
    target = tmp_path / 'workspace'
    formal_copy(target, benchmark='dexjoco', claude=True, agent_policy=True, experts=experts, task='click_mouse')
    agents = (target / 'AGENTS.md').read_text()
    assert 'You are the policy for one live episode' in agents
    for stale in ('explored this task', 'carried code', 'memory/PROGRESS.md', 'experience-memory'):
        assert stale not in agents
    assert ('runtime/experts/' in agents) == bool(experts)
    if experts:
        assert _skill(target, 'learn-from-experts') is not None


def test_agent_policy_prompt_and_config(monkeypatch):
    from services.agent_formal.runner import POLICY_PROMPT, episode_prompt
    from scripts.render_workspaces import configuration
    monkeypatch.delenv('RAX_UNATTENDED', raising=False)
    config = configuration(Path('/tmp/x'), agent_formal=True, agent_policy=True, formal_context='clear',
                           transfer_parts=())
    assert episode_prompt(config, []) == POLICY_PROMPT
    with pytest.raises(ValueError, match='agent_policy'):
        configuration(Path('/tmp/y'), agent_formal=True, agent_policy=True)  # carries the default compact context
