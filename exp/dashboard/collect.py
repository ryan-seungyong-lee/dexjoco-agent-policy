#!/usr/bin/env python3
"""Turn experiment sessions into dashboard data: per arm, per episode, what reached the agent
and what it did, with success, steps, time and tokens.

Every session with an exp_manifest.json under --sessions becomes <out>/<arm_id>/:
  session.json          the manifest, the arm's totals, and one entry per episode:
                          agp: each formal episode a live agent played (success, steps, minutes,
                               tokens, model requests, input exposure, its timeline)
                          cap: the exploration (one agent conversation over its episodes), the
                               milestones (first rehearsal that qualified, submission) with the
                               time and tokens spent to reach them, the formal score per episode
                               of the submitted program, and that controller.py
  timeline/<key>.jsonl  time-ordered events of one agent conversation: thinking, text, tool calls,
                        tool results (with the images the model saw), program-made robot calls
                        and touches of the demonstrations (runtime/experts)
  img/                  JPEG thumbnails (long side 384 px), named by content hash
and <out>/index.json lists every arm with its totals.

  python exp/dashboard/collect.py --sessions <RAX_EXP_ROOT>/runtime/auto-research --out exp/dashboard/site/data
"""
import argparse
import base64
import hashlib
import io
import json
import re
import shutil
from pathlib import Path

THUMB = 384
TEXT_LIMIT = 4000
EXPERT_PATTERN = re.compile(r'runtime/experts[^\s"\'`)]*')
USAGE_KEYS = ('input_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens', 'output_tokens')


# ------------------------------------------------------------------ helpers
def read_jsonl(path):
    rows = []
    try:
        with open(path, encoding='utf-8') as stream:
            for line in stream:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return rows


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def clip(value, limit=TEXT_LIMIT):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + f' … [+{len(text) - limit} chars]'


def iso_to_unix(value):
    from datetime import datetime
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    except (AttributeError, ValueError):
        return None


class Images:
    def __init__(self, directory):
        self.directory = directory

    def add(self, data):
        if not data:
            return None
        from PIL import Image
        raw = base64.b64decode(data) if isinstance(data, str) else data
        digest = hashlib.sha1(raw).hexdigest()[:16]
        path = self.directory / f'{digest}.jpg'
        if not path.exists():
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                image = Image.open(io.BytesIO(raw))
                image.thumbnail((THUMB, THUMB))
                image.convert('RGB').save(path, quality=80)
            except Exception:
                return None
        return f'img/{digest}.jpg'


def describe_observation(value):
    """Which inputs a robot tool reply carried, for the exposure counts."""
    if not isinstance(value, dict):
        return {}
    attachments = value.get('attachments') or []
    return {'rgb': sum(1 for a in attachments if a.get('kind') == 'rgb'),
            'depth': sum(1 for a in attachments if a.get('kind') == 'depth'),
            'camera_params': 'camera_intrinsics' in value or 'camera_poses' in value}


# ---------------------------------------------------------- one conversation
def transcript_events(paths, images):
    """Events of Claude Code transcripts, and their token usage (each model response counted once)."""
    events, usage, seen = [], dict.fromkeys(USAGE_KEYS, 0), set()
    usage['requests'] = 0
    for path in paths:
        sidechain_file = 'subagents' in Path(path).parts
        for row in read_jsonl(path):
            kind, message = row.get('type'), row.get('message') or {}
            content = message.get('content')
            base = {'t': iso_to_unix(row.get('timestamp', '')), 'src': 'agent',
                    'sidechain': bool(row.get('isSidechain') or sidechain_file)}
            if kind == 'system' and row.get('subtype') == 'compact_boundary':
                events.append({**base, 'kind': 'compaction'})
            elif kind == 'assistant' and isinstance(content, list):
                if message.get('id') not in seen and message.get('usage'):
                    seen.add(message.get('id'))
                    usage['requests'] += 1
                    for key in USAGE_KEYS:
                        usage[key] += message['usage'].get(key) or 0
                    base['cumulative_output'] = usage['output_tokens']
                for block in content:
                    btype = block.get('type')
                    if btype == 'thinking':
                        events.append({**base, 'kind': 'thinking', 'text': clip(block.get('thinking', ''))})
                    elif btype == 'text':
                        events.append({**base, 'kind': 'text', 'text': clip(block.get('text', ''))})
                    elif btype == 'tool_use':
                        arguments = block.get('input') or {}
                        event = {**base, 'kind': 'tool_use', 'id': block.get('id'),
                                 'tool': (block.get('name') or '').split('__')[-1], 'args': clip(arguments, 2000)}
                        demo = EXPERT_PATTERN.findall(json.dumps(arguments))
                        if demo:
                            event['demo_paths'] = sorted(set(demo))
                        events.append(event)
            elif kind == 'user':
                if isinstance(content, str):
                    events.append({**base, 'kind': 'user', 'text': clip(content)})
                    continue
                for block in content if isinstance(content, list) else []:
                    if block.get('type') == 'tool_result':
                        inner = block.get('content')
                        texts = ([b.get('text', '') for b in inner if b.get('type') == 'text']
                                 if isinstance(inner, list) else [inner or ''])
                        shown = [p for p in (images.add((b.get('source') or {}).get('data'))
                                             for b in (inner if isinstance(inner, list) else [])
                                             if b.get('type') == 'image') if p]
                        event = {**base, 'kind': 'tool_result', 'id': block.get('tool_use_id'),
                                 'text': clip('\n'.join(map(str, texts))), 'images': shown,
                                 'is_error': bool(block.get('is_error'))}
                        try:
                            event['inputs'] = describe_observation(json.loads(texts[0])) if texts else {}
                        except (ValueError, TypeError):
                            pass
                        events.append(event)
                    elif block.get('type') == 'text':
                        events.append({**base, 'kind': 'user', 'text': clip(block.get('text', ''))})
    uses = {e['id']: e for e in events if e['kind'] == 'tool_use'}
    for event in events:  # name each result after its call
        if event['kind'] == 'tool_result' and event.get('id') in uses:
            event['tool'] = uses[event['id']]['tool']
            if uses[event['id']].get('demo_paths'):
                event['demo_paths'] = uses[event['id']]['demo_paths']
    return events, usage


def mcp_events(path, agent_calls):
    """Robot and lifecycle calls from an MCP log. Each is matched, in order, to the agent's next
    unmatched call of the same tool made at most 60 s before it (the transcript stamps a call when the
    model emits it, the log when the server receives it); an unmatched call came from the agent's own
    code (a script or a controller)."""
    events, pending, unmatched = [], {}, sorted(agent_calls)
    for row in read_jsonl(path):
        kind = row.get('event')
        if kind == 'mcp_request':
            pending[row.get('call_id') or row.get('request_id')] = row
            continue
        if kind not in ('mcp_result', 'mcp_error'):
            continue
        request = pending.pop(row.get('call_id') or row.get('request_id'), {})
        t0, tool = request.get('time', row.get('time')), row.get('tool')
        if str(request.get('request_id') or row.get('request_id', '')).startswith(('supervisor', 'server-')):
            continue  # the harness's own observations and the client's tool discovery
        returned = row.get('returned') or {}
        if not returned and row.get('result'):
            content = (row['result'].get('content') or [{}])
            try:
                returned = json.loads(content[0].get('text', '{}'))
            except (ValueError, AttributeError):
                returned = {}
        match = next((i for i, (t, n, _) in enumerate(unmatched) if n == tool and t0 - 60 <= t <= t0 + 2), None)
        caller = 'agent' if match is not None else 'program'
        call_id = unmatched.pop(match)[2] if match is not None else None
        event = {'t': t0, 'src': 'mcp', 'kind': kind, 'tool': tool, 'caller': caller, 'call_id': call_id,
                 'args': clip(request.get('arguments'), 1500)}
        if isinstance(returned, dict):
            for key in ('step_id', 'task_complete', 'episode_id', 'bundle', 'rehearsal_qualified', 'error'):
                if key in returned:
                    event[key] = returned[key]
            if isinstance(returned.get('motion'), dict):
                event['motion'] = returned['motion']
        if kind == 'mcp_error':
            event['error'] = clip(row.get('error') or row, 500)
        events.append(event)
    return events


def timeline(transcripts, mcp_log, images):
    agent, usage = transcript_events(transcripts, images)
    calls = [(e['t'], e['tool'], e.get('id')) for e in agent if e['kind'] == 'tool_use' and e.get('t')]
    robot = mcp_events(mcp_log, calls) if mcp_log else []
    # The agent's own robot calls show as its tool results: give those the step and motion outcome.
    outcome = {e['call_id']: e for e in robot if e.get('call_id')}
    for event in agent:
        if event['kind'] == 'tool_result' and event.get('id') in outcome:
            for key in ('step_id', 'motion', 'task_complete'):
                if key in outcome[event['id']]:
                    event[key] = outcome[event['id']][key]
    events = sorted(agent + robot, key=lambda e: e.get('t') or 0)
    t0 = next((e['t'] for e in events if e.get('t')), None)
    for event in events:
        if t0 and event.get('t'):
            event['rel'] = round(event['t'] - t0, 1)
    exposure = {'rgb_images': 0, 'depth_images': 0, 'camera_params': 0, 'images_shown': 0, 'demo_reads': 0,
                'agent_tool_calls': 0, 'program_calls': 0}
    for event in events:
        if event['src'] == 'agent' and event['kind'] == 'tool_result':
            inputs = event.get('inputs') or {}
            exposure['rgb_images'] += inputs.get('rgb', 0)
            exposure['depth_images'] += inputs.get('depth', 0)
            exposure['camera_params'] += int(bool(inputs.get('camera_params')))
            exposure['images_shown'] += len(event.get('images') or [])
        if event['src'] == 'agent' and event['kind'] == 'tool_use':
            exposure['agent_tool_calls'] += 1
            exposure['demo_reads'] += int(bool(event.get('demo_paths')))
        if event['src'] == 'mcp' and event['caller'] == 'program':
            exposure['program_calls'] += 1
    return events, usage, exposure


def write_timeline(target, key, events):
    (target / 'timeline').mkdir(parents=True, exist_ok=True)
    with open(target / 'timeline' / f'{key}.jsonl', 'w') as stream:
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False, default=str) + '\n')
    return f'timeline/{key}.jsonl'


def tokens(usage):
    """input = fresh + cache read + cache write (what the model read), as Claude Code bills it."""
    if not usage:
        return None
    fresh = usage.get('input_tokens', 0)
    cached = usage.get('cache_read_input_tokens', usage.get('cached_input_tokens', 0)) or 0
    written = usage.get('cache_creation_input_tokens', usage.get('cache_write_input_tokens', 0)) or 0
    if 'cached_input_tokens' in usage:  # a result.json: input_tokens already includes the cache
        fresh = fresh - cached - written
    output = usage.get('output_tokens', 0) or 0
    return {'input': fresh + cached + written, 'cache_read': cached, 'cache_write': written,
            'fresh_input': fresh, 'output': output, 'total': fresh + cached + written + output}


def add_tokens(rows):
    total = {}
    for row in rows:
        for key, value in (row or {}).items():
            total[key] = total.get(key, 0) + (value or 0)
    return total or None


# -------------------------------------------------------------------- modes
def agent_policy_episodes(session, target, images):
    """agp: one entry per formal episode, played by its own live agent."""
    episodes = []
    for result_path in sorted(session.glob('published/runs/formal-*/result.json')):
        run = result_path.parent
        result = read_json(result_path, {})
        copy = run.name.removeprefix('formal-')
        transcripts = sorted(session.glob(f'agent-formal/episode-*/rax-controller-{copy}/claude-home/projects/**/*.jsonl'))
        events, usage, exposure = timeline(transcripts, run / 'mcp.jsonl', images)
        # The official evaluation layout this copy played (episode ids read <task>_formal_<layout>_<id>).
        found = re.search(r'_formal_(\d+)_', str(result.get('episode_id', '')))
        index = int(found.group(1)) + 1 if found else result.get('layout_id')
        key = f'formal-{index:03d}' if isinstance(index, int) else run.name
        timing = result.get('timing') or {}
        episodes.append({
            'key': key, 'kind': 'agent', 'label': f'평가 에피소드 {index}', 'layout_id': result.get('layout_id'),
            'success': result.get('task_complete'), 'end_reason': result.get('reason'),
            'steps': result.get('end_step_id'), 'minutes': round(timing.get('total_seconds', 0) / 60, 1) or None,
            'tokens': tokens(result.get('agent_token_usage') or usage),
            'model_requests': result.get('agent_model_requests', usage.get('requests')),
            'usage_limited': 'limit' in json.dumps(result.get('agent_errors') or []),
            'agent_errors': [clip(e.get('error'), 300) for e in result.get('agent_errors') or []],
            'exposure': exposure, 'timeline': write_timeline(target, key, events),
            'final_text': next((e['text'] for e in reversed(events) if e['kind'] == 'text'), None)})
    return sorted(episodes, key=lambda e: e['key'])


def code_policy(session, target, images):
    """cap: the exploration conversation, its milestones, and the formal scores of its program."""
    transcripts = sorted((session / 'claude-home' / 'projects').rglob('*.jsonl'))
    events, usage, exposure = timeline(transcripts, session / 'mcp.jsonl', images)
    t0 = next((e['t'] for e in events if e.get('t')), None)

    def milestone(predicate):
        hit = next((e for e in events if e['src'] == 'mcp' and predicate(e)), None)
        if not hit or not t0:
            return None
        spent = [e.get('cumulative_output') for e in events if e.get('t') and e['t'] <= hit['t'] and e.get('cumulative_output')]
        return {'minutes': round((hit['t'] - t0) / 60, 1), 'output_tokens': spent[-1] if spent else None}

    exploration = [e for e in events if e['src'] == 'mcp' and e['tool'] == 'start_episode' and e['kind'] == 'mcp_result']
    scores = read_json(session / 'formal-scores.json') or {}
    formal = [{'key': f"program-{e.get('episode_index')}", 'kind': 'program',
               'label': f"평가 에피소드 {e.get('episode_index')}", 'success': bool(e.get('native_complete')),
               'steps': e.get('step_id')} for e in scores.get('episodes', [])]
    controller = sorted(session.glob('bundles/*/controller.py')) or sorted(session.glob('**/code/*/controller.py'))
    return {
        'exploration': {'key': 'exploration', 'kind': 'exploration', 'label': '탐색 (코드 작성)',
                        'episodes_used': len(exploration), 'tokens': tokens(usage),
                        'model_requests': usage.get('requests'), 'exposure': exposure,
                        'minutes': round((events[-1]['t'] - t0) / 60, 1) if t0 and events[-1].get('t') else None,
                        'timeline': write_timeline(target, 'exploration', events)},
        'milestones': {
            'first_qualified_rehearsal': milestone(lambda e: e['tool'] == 'rehearse' and e.get('rehearsal_qualified')),
            'submission': milestone(lambda e: e['tool'] == 'submit' and e['kind'] == 'mcp_result')},
        'formal': formal,
        'controller_py': clip(controller[-1].read_text(errors='replace'), 60000) if controller else None}


def collect(session, out):
    manifest = read_json(session / 'exp_manifest.json')
    if not manifest:
        return None
    arm = manifest['arm_id']
    target = out / arm
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    images = Images(target / 'img')
    mode = manifest.get('mode', 'cap')
    summary = {'manifest': manifest, 'session': session.name, 'mode': mode}
    if mode == 'agp':
        episodes = agent_policy_episodes(session, target, images)
        valid = [e for e in episodes if not e['usage_limited']]
        summary.update(episodes=episodes, tokens=add_tokens(e['tokens'] for e in valid))
    else:
        cap = code_policy(session, target, images)
        valid = cap['formal']
        summary.update(cap, episodes=cap['formal'], tokens=cap['exploration']['tokens'])
    summary.update(
        evaluated=len(valid), successes=sum(bool(e['success']) for e in valid),
        success_rate=(sum(bool(e['success']) for e in valid) / len(valid)) if valid else None,
        usage_limited=sum(bool(e.get('usage_limited')) for e in summary['episodes']),
        minutes_per_episode=(round(sum(e.get('minutes') or 0 for e in valid) / len(valid), 1)
                             if valid and mode == 'agp' else None))
    (target / 'session.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', type=Path, required=True, help='<RAX_EXP_ROOT>/runtime/auto-research')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arms', nargs='*', help='only these arm ids (default: every arm)')
    parser.add_argument('--arms-dir', type=Path,
                        help='<RAX_EXP_ROOT>/arms: collect only the session each arm currently points to '
                             '(arms/<id>/config_path), so superseded or invalidated reruns stay out')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / 'index.json'
    # A partial update (--arms) keeps the other arms; a full one rebuilds the index.
    index = {row['arm_id']: row for row in read_json(index_path, [])} if args.arms else {}
    if args.arms_dir:
        sessions = [Path(p.read_text().strip()).parent for p in sorted(args.arms_dir.glob('*/config_path'))
                    if '.' not in p.parent.name]
    else:
        sessions = sorted(p for p in args.sessions.iterdir() if p.is_dir())
    for session in sessions:
        manifest = read_json(session / 'exp_manifest.json')
        if not manifest or (args.arms and manifest['arm_id'] not in args.arms):
            continue
        summary = collect(session, args.out)
        m = summary['manifest']
        index[m['arm_id']] = {
            'arm_id': m['arm_id'], 'phase': m['arm_id'].split(f"_{m['task']}_")[0], 'mode': summary['mode'],
            **{k: m.get(k) for k in ('task', 'profile', 'expert', 'model', 'agent', 'effort', 'rep')},
            **{k: summary.get(k) for k in ('evaluated', 'successes', 'success_rate', 'usage_limited',
                                           'minutes_per_episode', 'tokens')}}
        print('collected', m['arm_id'], f"{summary['successes']}/{summary['evaluated']}")
    index_path.write_text(json.dumps(sorted(index.values(), key=lambda r: r['arm_id']), indent=2))
    print(len(index), 'arms ->', args.out)


if __name__ == '__main__':
    main()
