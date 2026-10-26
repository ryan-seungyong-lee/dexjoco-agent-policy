"""The workspace recipe for each role: an exploring agent and an agent formal copy."""
from dataclasses import replace

from .recipe import PROJECT, Recipe, SkillRule, Vocab
from .setup import wrist_cameras

VOCAB = Vocab(memory='memory', notes='memory/PROGRESS.md', frames='runtime/autonomous_controller/runs/',
              evaluate='evaluate', episode_modes='exploration or rehearsal')

# Skill sources by axis. Every workspace takes the task and coordination skills, the
# perception skills of its observation mode ('depth' or 'rgb-only') and robot, the
# control skills of its robot (named by its benchmark in services/benchmarks.py), and the
# auto-research skills. Agents see one flat set with stable, robot-neutral names
# (robot-interface, free-space-move, object-localization, ...), so skills link to each
# other without branching on the robot or the observation mode.
# Benchmarks with method_skills also take the robot-neutral research-method skills
# ('method': evidence, regression, controller engineering, rehearsal, budgets).
SKILL_LAYERS = ('task', 'coordination', 'perception/common', 'perception/{perception}',
                'perception/{perception}-{robot}', 'control/common', 'control/{robot}', 'auto-research',
                '{method}')

# Skills a setup does not use.
COMMON_OMISSIONS = (
    # Single-environment sessions have no env_id routing.
    SkillRule('multi-env-exploration', lambda s: s.contract.environments == 1),
    SkillRule('multi-agent-exploration', lambda s: not s.contract.subagent_cap),
    # Visual demonstrations are optional context; without them the skill has nothing to read.
    SkillRule('in-context-action-learning', lambda s: not s.demonstrations),
    # Complete expert episodes are optional context too.
    SkillRule('learn-from-experts', lambda s: not s.expert_demonstrations or s.expert_content.startswith('human-')),
    # Human demonstrations replace the robot experts.
    SkillRule('learn-from-human-demos', lambda s: not s.expert_demonstrations
              or not s.expert_content.startswith('human-')),
    # Moving a wrist camera closer needs a robot with wrist cameras.
    SkillRule('wrist-camera-inspection', lambda s: not wrist_cameras(s.contract.benchmark)),
    # Target-level motion has no row preview.
    SkillRule('motion-preview', lambda s: s.contract.action_interface == 'targets'),
)
# Method skills an agent formal copy has no use for: it plays one episode live, with no
# later episodes to compare, no bundle to rehearse and no changelog to keep.
FORMAL_METHOD_OMISSIONS = tuple(SkillRule(name, lambda s: True) for name in (
    'avoid-overfitting', 'change-discipline', 'engineer-the-controller', 'evidence-and-regression',
    'learn-from-incidents', 'prioritize-by-funnel', 'rehearse-the-controller'))
# Collection references an agent formal copy has no use for.
FORMAL_REMOVED = ('experience-memory/references/format.md', 'robot-interface/references/pose-records.md')

AUTO_RESEARCH = Recipe('auto-research/exploration', PROJECT / 'auto_research_agent', SKILL_LAYERS,
                       COMMON_OMISSIONS + (
                           # handoff/ is written by an exploring agent whose work a later copy
                           # continues: live on the held-out layouts (when they carry it), or in an
                           # adaptation trial.
                           SkillRule('transfer-handoff', lambda s: 'formal-episode' in s.options
                                     or not ('handoff' in s.options or 'agent-formal' in s.options
                                             and s.carries_handoff)),
                           # Live-copy mode has no bundle to rehearse.
                           SkillRule('rehearse-the-controller', lambda s: 'agent-formal' in s.options),),
                       role_documents=('FORMAL_AGENTS.md',), vocab=VOCAB)
# An agent formal copy: one live held-out episode, its own instructions, no bundle to
# register, rehearse or submit, and formal-copy variants of the memory and
# autonomous-control (supervised automation) skills.
# Agent as policy carries nothing: no code to run under supervision, no tips to reuse.
AGENT_POLICY_OMISSIONS = tuple(SkillRule(name, lambda s: 'agent-policy' in s.options)
                               for name in ('autonomous-control', 'experience-memory'))
FORMAL_COPY = replace(AUTO_RESEARCH, name='auto-research/formal-copy', agents_source='FORMAL_AGENTS.md',
                      omit_skills=AUTO_RESEARCH.omit_skills + FORMAL_METHOD_OMISSIONS + AGENT_POLICY_OMISSIONS,
                      drop_documents=('START_PROMPT.md',), role_documents=(),
                      variant='formal-episode', remove_files=FORMAL_REMOVED, options=frozenset({'formal-episode'}))


def recipe_for(options=frozenset()):
    """The role's recipe; the robot and observation mode come from the setup at build time."""
    return FORMAL_COPY if 'formal-episode' in options else AUTO_RESEARCH
