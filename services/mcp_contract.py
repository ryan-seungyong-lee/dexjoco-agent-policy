"""The auto-research session's capability contract (robot and lifecycle tools).

This module is pure configuration: no simulator, process-global mode, or secrets.
Discovery and dispatch use the same immutable session contract.
"""
from copy import deepcopy
from dataclasses import dataclass

from services.benchmarks import ALL_ROBOT_TOOLS, BENCHMARKS, DEFAULT

GEOMETRY_TOOLS = frozenset().union(*(b.geometry_tools for b in BENCHMARKS.values()))
# rgbd: RGB + depth + camera calibration; rgb-depth: RGB + depth, no calibration;
# rgb-camera: RGB + calibration (intrinsics, poses), no depth; rgb-only: RGB alone.
PROFILES = ('rgbd', 'rgb-depth', 'rgb-camera', 'rgb-only')


# Session lifecycle tools in exploration, and those of them scoped to one environment.
LIFECYCLE_TOOLS = frozenset({'status', 'start_episode', 'evaluate', 'finish', 'register', 'rehearse', 'submit'})
# Code mode's evaluation tools; live-copy mode (agent formal) has no bundle evaluation.
BUNDLE_TOOLS = frozenset({'register', 'rehearse', 'submit'})
EPISODE_TOOLS = frozenset({'start_episode', 'evaluate', 'finish'})
# Recorded in MCP_CONTRACT.json; auto-research is the only experiment mode.
MODE = 'auto-research'
DEPTH_KEYS = frozenset({
    'depth',
    # Every registered benchmark camera's depth array (<camera>_depth_m).
    *(f'{camera}_depth_m' for bench in BENCHMARKS.values() for camera in bench.robot.get('cameras', ())),
})
CAMERA_KEYS = frozenset({'camera_parameters', 'intrinsic_matrix', 'extrinsic_matrix', 'camera_to_world_usd',
                         'camera_poses', 'camera_intrinsics'})
GEOMETRY_KEYS = DEPTH_KEYS | CAMERA_KEYS


@dataclass(frozen=True)
class ObservationProfile:
    name: str = 'rgbd'

    def __post_init__(self):
        if self.name not in PROFILES:
            raise ValueError(f"observation_profile must be one of {', '.join(PROFILES)}")

    @property
    def depth(self):
        """Depth images (and their stats) reach the agent."""
        return self.name in ('rgbd', 'rgb-depth')

    @property
    def camera_params(self):
        """Camera intrinsics and poses reach the agent."""
        return self.name in ('rgbd', 'rgb-camera')

    @property
    def geometry(self):
        """Depth and calibration together: what the pixel-to-position tools need."""
        return self.depth and self.camera_params

    def tools(self, definitions):
        tools = deepcopy(definitions)
        if self.geometry:
            return tools
        result = []
        for tool in tools:
            if tool['name'] in GEOMETRY_TOOLS:
                continue
            description = tool.get('description', '')
            if not self.depth:
                description = description.replace('RGB/depth', 'RGB').replace(
                    'RGB (640x640) and depth attachments', 'RGB (640x640) attachments')
            if self.depth and not self.camera_params:
                description = description.replace('attachments, calibration,', 'attachments (no calibration),')
            elif self.camera_params and not self.depth:
                description = description.replace('attachments, calibration,',
                                                  'attachments (no depth), calibration,')
            else:
                description = description.replace('attachments, calibration,',
                                                  'attachments (no depth or calibration),')
            tool['description'] = description
            result.append(tool)
        return result

    def stripped_keys(self):
        return (frozenset() if self.depth else DEPTH_KEYS) | (frozenset() if self.camera_params else CAMERA_KEYS)

    def clean(self, value):
        if self.geometry:
            return value
        return self._clean(value, self.stripped_keys())

    def _clean(self, value, keys):
        if isinstance(value, dict):
            return {key: self._clean(item, keys) for key, item in value.items()
                    if str(key).lower() not in keys}
        if isinstance(value, (list, tuple)):
            return [self._clean(item, keys) for item in value
                    if self.depth or not (isinstance(item, dict) and item.get('kind') == 'depth')]
        return value


@dataclass(frozen=True)
class Contract:
    phase: str = 'exploration'
    observation_profile: str = 'rgbd'
    environments: int = 1
    # Concurrent subagents allowed in exploration. 0 = automatic: one per
    # environment when there are several, none with a single environment.
    subagents: int = 0
    benchmark: str = DEFAULT
    # Code mode (the default): exploration registers, rehearses and submits a bundle.
    # Live-copy mode (agent formal) evaluates live copies of the agent instead, so
    # exploration has no bundle tools.
    bundle_tools: bool = True
    external_perception: bool = True
    # 'rows': the benchmark's official per-step action rows (RAX). 'targets': agent as policy's
    # target-level motion tools instead (DexJoCo: dexjoco_move_ee and dexjoco_hand).
    action_interface: str = 'rows'

    def __post_init__(self):
        if self.benchmark not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark; choose one of {', '.join(sorted(BENCHMARKS))}")
        if self.phase not in {'exploration', 'isolated'}:
            raise ValueError('Unknown MCP phase')
        ObservationProfile(self.observation_profile)
        if type(self.environments) is not int or not 1 <= self.environments <= 8:
            raise ValueError('environments must be an integer in 1..8')
        if self.environments != 1 and self.phase != 'exploration':
            raise ValueError('Multiple environments are exploration-only')
        if type(self.subagents) is not int or not 0 <= self.subagents <= 8:
            raise ValueError('subagents must be an integer in 0..8')
        if self.subagents and self.phase != 'exploration':
            raise ValueError('Subagents are exploration-only')
        if type(self.bundle_tools) is not bool:
            raise ValueError('bundle_tools must be a boolean')
        if type(self.external_perception) is not bool:
            raise ValueError('external_perception must be a boolean')
        if self.action_interface not in ('rows', 'targets'):
            raise ValueError("action_interface must be 'rows' or 'targets'")
        if self.action_interface == 'targets' and not self.bench.target_tools:
            raise ValueError(f'{self.bench.label} has no target-level motion tools')

    @classmethod
    def from_config(cls, config, *, isolated=False):
        return cls('isolated' if isolated else 'exploration',
                   getattr(config, 'observation_profile', 'rgbd'),
                   1 if isolated else getattr(config, 'exploration_envs', 1),
                   0 if isolated else getattr(config, 'subagents', 0),
                   getattr(config, 'benchmark', DEFAULT),
                   isolated or not getattr(config, 'agent_formal', False),
                   getattr(config, 'external_perception', True),
                   'targets' if getattr(config, 'agent_policy', False) else 'rows')

    @property
    def subagent_cap(self):
        """Maximum concurrent subagents; 0 means the agent works alone."""
        return self.subagents or (self.environments if self.environments > 1 else 0)

    @property
    def observation(self):
        return ObservationProfile(self.observation_profile)

    @property
    def bench(self):
        return BENCHMARKS[self.benchmark]

    @property
    def robot_names(self):
        bench = self.bench
        tools = bench.robot_tools - (bench.row_tools if self.action_interface == 'targets' else bench.target_tools)
        return tools if self.observation.geometry else tools - self.bench.geometry_tools

    @property
    def scoped_tools(self):
        return (self.robot_names - self.bench.unscoped_tools) | EPISODE_TOOLS

    def select(self, definitions, *, scoped=None):
        scoped = self.scoped_tools if scoped is None else scoped
        tools = self.observation.tools(definitions)
        selected = []
        allowed = self.tool_names
        for tool in tools:
            name = tool['name']
            if name not in allowed:
                continue
            if name in scoped and self.environments > 1:
                schema = tool['inputSchema']
                schema.setdefault('properties', {})['env_id'] = {
                    'type': 'integer', 'minimum': 0, 'maximum': self.environments - 1,
                    'description': 'Exploration environment ID; calls serialize within each environment.'}
                if 'env_id' not in schema.setdefault('required', []):
                    schema['required'].append('env_id')
                if name == 'start_episode':
                    tool['description'] = tool['description'].replace('closes the previous one',
                        'closes the previous episode in this environment only')
                suffix = (' Select env_id explicitly; observations and cached plans must belong to this environment '
                          'and episode. All environments share the exploration episode budget.')
                if not tool['description'].endswith(suffix):
                    tool['description'] += suffix
            selected.append(tool)
        return selected

    def robot_definitions(self):
        return self.select(self.bench.definitions(), scoped=self.robot_names - self.bench.unscoped_tools)

    def require_robot(self, name, arguments):
        if name not in self.robot_names and (name != 'gemini_generate' or not self.external_perception):
            raise PermissionError('Tool is not available in this MCP contract')
        if not isinstance(arguments, dict):
            raise ValueError('Arguments must be an object')
        scoped = name in self.robot_names - self.bench.unscoped_tools
        if scoped and self.environments > 1:
            env_id = arguments.get('env_id')
            if type(env_id) is not int or not 0 <= env_id < self.environments:
                raise ValueError('A valid explicit env_id is required')
        elif 'env_id' in arguments:
            raise ValueError('env_id is not available for this tool/context')

    def route(self, name, arguments):
        """Validate and remove the host routing field before a native worker call."""
        if name not in self.scoped_tools:
            raise PermissionError('Tool is not available as an environment-scoped call')
        if not isinstance(arguments, dict):
            raise ValueError('Arguments must be an object')
        payload = dict(arguments)
        if self.environments == 1:
            if 'env_id' in payload:
                raise ValueError('env_id is not available in a single environment')
            return 0, payload
        env_id = payload.pop('env_id', None)
        if type(env_id) is not int or not 0 <= env_id < self.environments:
            raise ValueError('A valid explicit env_id is required')
        return env_id, payload

    @property
    def lifecycle_tools(self):
        if self.phase != 'exploration':
            return frozenset()
        return LIFECYCLE_TOOLS if self.bundle_tools else LIFECYCLE_TOOLS - BUNDLE_TOOLS

    @property
    def tool_names(self):
        """Every MCP tool this contract exposes."""
        return frozenset(self.robot_names | self.lifecycle_tools |
                         ({'gemini_generate'} if self.external_perception else set()))

    def manifest(self):
        manifest = {'mode': MODE, 'phase': self.phase, 'observation_profile': self.observation_profile,
                    'exploration_envs': self.environments, 'subagents': self.subagent_cap, 'robot_tools': sorted(self.robot_names),
                    'gemini': self.external_perception, 'benchmark': self.benchmark}
        if self.action_interface != 'rows':
            manifest['action_interface'] = self.action_interface
        return manifest

def observation_for(service):
    """Native services default to RGBD; launchers supply an immutable profile."""
    return ObservationProfile(getattr(service, 'observation_profile', 'rgbd'))
