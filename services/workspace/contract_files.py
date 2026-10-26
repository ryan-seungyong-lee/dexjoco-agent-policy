"""MCP_CONTRACT.json and MCP_SESSION.md: the session's tool contract in the workspace."""
import json
from pathlib import Path

from services.mcp_contract import Contract


def describe(contract: Contract, options=frozenset()) -> str:
    localization = "[object-localization](.agents/skills/object-localization/SKILL.md)"
    if contract.observation.geometry:
        geometry = f"RGB images, depth and camera calibration are supplied; get metric positions with {localization}."
    elif contract.observation.depth:
        geometry = (f"RGB images and depth images are supplied, but no camera calibration (intrinsics or poses); "
                    f"locate objects with {localization}.")
    elif contract.observation.camera_params:
        geometry = (f"RGB images and camera calibration (intrinsics and world poses) are supplied, but no depth; "
                    f"locate objects with {localization}.")
    else:
        geometry = f"Only RGB images and measured robot state are supplied; locate objects with {localization}."
    environments = ''
    if contract.environments > 1:
        environments = (f"Exploration has {contract.environments} environments; environment-scoped tools take "
                        "`env_id`; see [multi-env-exploration](.agents/skills/multi-env-exploration/SKILL.md). "
                        f"Up to {contract.subagent_cap} subagents may run at once.")
    elif contract.subagent_cap:
        environments = (f"Up to {contract.subagent_cap} subagents may run at once; see "
                        "[multi-agent-exploration](.agents/skills/multi-agent-exploration/SKILL.md).")
    role_line = {'live-copy': 'one live episode',
                 'formal-worker': 'one held-out formal episode',
                 'exploration': 'exploration'}[role(contract, options)]
    tools = sorted(contract.tool_names | ({'status'} if 'formal-episode' in options else set()))
    listing = ("Tools: " + ", ".join(f"`{name}`" for name in tools)
               + ". The live tool schemas give arguments and limits.")
    robot_status = contract.bench.status_tool
    if 'status' in tools and robot_status in tools:
        reports = ("this episode's limits" if 'agent-policy' in options else
                   "this episode's limits and what you carried" if 'formal-episode' in options else
                   "budgets, bundles and episode state" if contract.bundle_tools else
                   "budgets, episode state and the carried folders")
        listing += f" `status` reports {reports}; `{robot_status}` reports the live robot and episode."

    return (f"# Session tools\n\nRole: {role_line}. "
            f"Observation profile: {contract.observation_profile}.\n\n"
            + "\n\n".join(part for part in (geometry, environments, listing) if part) + "\n")


def role(contract, options=frozenset()):
    """Which agent reads this workspace, beyond the MCP phase."""
    if 'formal-episode' in options:
        return 'live-copy'
    return 'formal-worker' if contract.phase == 'isolated' else 'exploration'


def manifest(contract, options=frozenset()):
    """The MCP contract plus this workspace's role and its exact tool list."""
    return {**contract.manifest(), 'role': role(contract, options),
            'tools': sorted(contract.tool_names | ({'status'} if 'formal-episode' in options else set()))}


def write_contract(workspace, contract, options=frozenset()):
    workspace = Path(workspace)
    (workspace / "MCP_CONTRACT.json").write_text(json.dumps(manifest(contract, options), indent=2) + "\n")
    (workspace / "MCP_SESSION.md").write_text(describe(contract, options))
