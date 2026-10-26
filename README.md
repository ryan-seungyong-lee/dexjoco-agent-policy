# Dex Agent Policy: LLM 에이전트로 dex hand 로봇 제어하기

LLM 에이전트(Claude Code, Codex)가 DexJoCo 시뮬레이터에서 Franka 팔 + 16자유도 Allegro 손을
제어할 때, **어떤 입력(RGB / depth / 카메라 파라미터)과 어떤 데모(로봇·사람)가 얼마나 도움이 되는지**
재는 실험 코드입니다. 에이전트가 정책이 되는 방식을 두 가지로 나눠 비교합니다.

- **Agent as policy (AgP)**: 에이전트가 평가 에피소드마다 직접 로봇을 관찰하고 조작합니다.
- **Code as policy (CaP)**: 에이전트가 연습하며 제어 프로그램을 짜고, 그 프로그램 혼자 평가받습니다.

[RAX](https://github.com/robot-agent-exploration/rax)의 `dexjoco/native` 브랜치(Claude Code / Codex
하네스, MCP 로봇 도구, 샌드박스, DexJoCo 백엔드)를 바탕으로 실험에 맞게 고쳤습니다. 원본 문서는
[docs/RAX_README.md](docs/RAX_README.md)와 `docs/`에 있습니다.

목차: [설치](#1-설치) · [로그인](#2-로그인-claude-code--codex) · [모드](#3-모드-agent-as-policy--code-as-policy) ·
[실행](#4-실행) · [대시보드](#5-대시보드) · [구조](#6-코드-구조와-rax에서-바뀐-점)

---

## 1. 설치

요구사항: Linux x86_64, EGL을 지원하는 NVIDIA GPU (**A100은 안 됩니다**: DexJoCo 장면의 MSAA가 실패합니다.
RTX/A6000/4090/H100은 됩니다), `bubblewrap`(+`socat`, Claude Code 샌드박스용), `git-lfs`, `nvidia-smi`.
Docker는 필요 없습니다. 자세한 내용은 [docs/SETUP.md](docs/SETUP.md).

```bash
bash scripts/bootstrap_sources.sh      # 고정된 DexJoCo 소스
bash scripts/setup_host_env.sh         # 하네스용 Python 환경 (runtime/envs/host)
bash scripts/setup_dexjoco_env.sh      # 시뮬레이터 환경
# 로봇 데모 (expert 조건에 필요): 시뮬레이터에서 재생해 성공하고, 공식 평가 배치와 겹치지 않는 것만 설치
runtime/envs/host/bin/python scripts/setup_dexjoco_demos.py --gpu 0 --task click_mouse --task hammer_nail
runtime/envs/host/bin/python exp/demos/robot_keyposes.py --gpu 0 --task click_mouse --task hammer_nail   # 데모 요약
bash scripts/doctor.sh && runtime/envs/host/bin/python scripts/dexjoco_doctor.py --gpu 0 --task click_mouse
```

- `bwrap`은 0.10 이상이어야 합니다 (평가 샌드박스가 `--size`를 씁니다). 노드에 0.9가 깔려 있다면
  conda-forge의 `bubblewrap`을 호스트 환경에 설치하세요: `exp/slurm/setup_cpu.sbatch`가 그렇게 합니다.
- 양재 클러스터에서는 위 단계를 잡으로 돌립니다: `exp/slurm/setup_cpu.sbatch`, `setup_gpu.sbatch`
  (로그인 노드에서는 설치나 에이전트를 돌리지 않습니다).
- 동작 확인: `exp/checks/target_motion_check.py`(AgP 이동 도구), `exp/checks/profile_check.py`(입력 4종).

## 2. 로그인 (Claude Code / Codex)

에이전트 CLI는 호스트에서 그대로 돌고, 세션마다 자기 CLI 홈을 따로 씁니다. 개인 설정과 기록은 쓰지 않습니다.

### Claude Code (검증됨)

1. Claude Code 설치 (테스트 버전 2.1.296).
2. **구독(Pro/Max)으로 쓰기**: 실험에 쓸 계정으로 로그인한 머신에서 장기 토큰을 발급해
   `~/.config/rax/claude-experiment.token`(권한 600)에 저장합니다.
   ```bash
   claude setup-token                       # 출력된 sk-ant-oat01-... 토큰을 복사
   mkdir -p ~/.config/rax && umask 077 && cat > ~/.config/rax/claude-experiment.token   # 붙여넣고 Enter, Ctrl-D
   ```
   클러스터에 둘 때는 노트북에서 `ssh <host> 'mkdir -p ~/.config/rax && umask 077 && cat > ~/.config/rax/claude-experiment.token'`.
   `scripts/setup_claude_experiment_login.sh`를 쓰면 실험 전용 계정 디렉터리까지 만들어 줍니다.
3. **API 키로 쓰기**: 키를 파일에 두고 `PROVIDER=anthropic KEY=<파일>`로 실행합니다.

- 세션이 고른 모델(`MODEL`)이 보조 요청과 subagent까지 모두 쓰도록 고정됩니다 (다른 모델이 섞이지 않음).
- 구독은 **세션 사용 한도**가 있습니다. 한도에 걸린 에피소드는 집계에서 빠지고(`usage-limited`),
  탐색 중인 에이전트는 리셋까지 기다렸다가 이어 갑니다. 한도를 피하려면 동시에 돌리는 arm 수를 줄이세요
  (`exp/launch.py --slots`, 기본 3). 같은 계정으로 다른 Claude 작업을 하면 한도를 같이 씁니다.

### Codex (문서만, 이 실험에서는 아직 미검증)

RAX의 기본 에이전트이고 하네스는 그대로 지원합니다. Codex CLI(테스트 버전 0.162.1)를 설치하고
provider profile을 `~/.codex/<profile>.config.toml`에 씁니다. 모델은 profile에서 정합니다.

```toml
# ChatGPT 구독: codex login 으로 만든 auth.json 을 지정
[rax]
chatgpt_auth = "auth.json"          # ~/.codex 기준 경로
model = "gpt-6-astra"
```

```toml
# OpenAI API 키 (Responses API)
model_provider = "openai"
[model_providers.openai]
name = "OpenAI"
base_url = "https://api.openai.com/v1"
wire_api = "responses"
[rax]
key_file = "secrets/openai_api_key"  # 권한 600
model = "gpt-6-astra"
native_models = true
```

실행할 때 `AGENT=codex CODEX_PROFILE=<profile>`을 줍니다. 대시보드의 대화 타임라인은 Claude Code
기록 형식만 읽습니다 (Codex는 로봇 호출과 결과만 보입니다).

## 3. 모드: Agent as policy / Code as policy

두 모드는 같은 시뮬레이터, 같은 입력 조건, 같은 데모를 쓰지만 **평가받는 대상**이 다릅니다.

| | Agent as policy (`MODE=agp`) | Code as policy (`MODE=cap`) |
|---|---|---|
| 평가받는 것 | 에이전트 자신 (평가 에피소드마다 새 에이전트가 직접 조작) | 에이전트가 쓴 `controller.py` (LLM 없이 혼자 실행) |
| 연습 | 없음 | 탐색 에피소드 `EPISODES`개 (기본 5)에서 직접 조작하며 코드 작성 → 리허설 → 제출 |
| 에피소드 간 기억 | 없음 (대화·메모·코드 아무것도 넘어가지 않음) | 탐색 내내 한 대화, 메모와 코드가 쌓임 |
| 로봇 도구 | 목표 단위: `dexjoco_move_ee`, `dexjoco_hand`, `dexjoco_hold` | 20 ms 단위 행: `dexjoco_step` (+ 미리보기) |
| 공통 도구 | 관찰 `dexjoco_observe`, 위치 측정 `dexjoco_pixel_to_position`(rgbd), 회전 계산, Python 스크립트 | 같음 |
| 평가 에피소드 | `FORMAL_EPISODES`개 (파일럿 3) | 공식 150개 |
| 주요 지표 | 성공률, 에피소드당 시간·토큰 | 첫 리허설 통과·제출까지 시간·토큰, 성공률 |

**AgP의 이동 도구** ([Agent-as-Policy](https://github.com/agent-as-policy-2026/agent-as-policy)의
`move_ee`를 따름, 손은 RAX DexJoCo 규약):

- `dexjoco_move_ee`: 팔 flange의 목표 자세(`position_m` 또는 `delta_m`, `rotation`)를 주면 서버가
  20 ms 행으로 보간(빠름 `transit` ~0.24 m/s, 정밀 `precision` ~0.075 m/s)하고, 공식 제어기로 실행한 뒤
  목표 근처에서 안정화합니다. 안정화 동안은 남은 오차의 1.5배만큼 앞질러 명령해(최대 3 cm) 느린 제어기를
  빨리 수렴시키고, 무언가에 막혀 멈추면 `stalled`로 일찍 끝냅니다. 도달 여부·오차·쓴 스텝을 돌려줍니다.
- `dexjoco_hand`: 16관절 목표로 손을 움직이고, 접촉 때문에 목표에 못 간 관절(`blocked_joints`)을 알려 줍니다.
- 바꾸지 않은 목표는 유지됩니다: 손을 쥔 채 팔을 옮기면 계속 쥐고 있습니다.

**두 모드 공통**

- 사람이 지켜보지 않는 실행입니다. 모든 프롬프트에 "운영자가 없으니 스스로 결정하라
  (CaP는 탐색이 끝나기 전에 가장 나은 controller를 제출하라)"는 안내가 붙습니다 (`RAX_UNATTENDED=1`).
- 외부 인식(Gemini)과 에피소드 리뷰는 끕니다: 모델 자신의 시각만 잽니다.

**입력 조건** (`PROFILE`): `rgb-only`, `rgb-depth`(depth만), `rgb-camera`(카메라 내·외부 파라미터만), `rgbd`(둘 다;
`pixel_to_position` 도구는 이때만 있음).
**데모** (`EXPERT`): `none`, `full`(로봇 데모 3개: 핵심 순간 요약 `keyposes.json` + 영상 + 상태/행동), `video`(로봇 영상만),
`human-video`, `human-full`(사람 영상 + 3D 손 궤적·접촉 단계·Allegro 손 모양 힌트; `exp/human_demos/`,
`exp/slurm/human_demos.sbatch`. EgoDex 데이터 경로를 `EGODEX_ROOT`, `RLHND_ROOT`로 줍니다).

## 4. 실행

### arm 하나 (GPU 머신 어디서나)

```bash
ARM_ID=try_click MODE=agp TASK=click_mouse PROFILE=rgbd EXPERT=none REP=0 FORMAL_EPISODES=3 \
MODEL=claude-sonnet-5-5 bash exp/run_arm.sh
```

| 변수 | 뜻 |
|---|---|
| `MODE` | `agp` 또는 `cap` |
| `AGENT`, `MODEL` | `claude` + Claude 모델 id (기본 `claude-sonnet-5-5`; `claude-haiku-5-5`, `claude-opus-5-5`) 또는 `codex` + `CODEX_PROFILE` |
| `EFFORT` | 추론 effort (`xhigh` 기본, `high`, `max`) |
| `TASK` | DexJoCo task (`click_mouse`, `hammer_nail`, `pick_bucket`, `water_plant`, ...) |
| `PROFILE`, `EXPERT` | 위의 입력 조건과 데모 |
| `EPISODES`, `FORMAL_EPISODES` | CaP 탐색 에피소드 수, 평가 에피소드 수 |
| `SIM_GPU` | 시뮬레이터 GPU UUID (기본: 보이는 첫 GPU) |

세션은 `runtime/auto-research/<id>/`에, arm 기록은 `$RAX_EXP_ROOT/arms/<ARM_ID>`(기본 `~/rax_haiku`)와
`ledger.jsonl`에 남습니다. 같은 `ARM_ID`로 다시 실행하면 그 세션을 이어 갑니다(선점된 잡의 재시작).

### 실험 매트릭스 (Slurm)

조건은 [`exp/arms.yaml`](exp/arms.yaml)의 phase로 정의합니다 (task × 반복 × arm, 모델·모드·에피소드 수는
phase나 arm마다 덮어쓰기 가능).

```bash
export RAX_SLURM_HOST=<ssh 별칭>
python3 exp/launch.py --phase sonnet_screen                # 제출 (--dry-run 으로 미리보기)
python3 exp/launch.py --phase model_opus --slots 3
ssh <host> 'python3 ~/rax_haiku/rax/exp/tally.py'          # arm별 성공/시간/토큰 요약
```

- 제출할 때 커밋된 HEAD를 클러스터의 `~/rax_haiku/releases/<sha>`로 얼려 두고 그 코드로 돌립니다.
  돌고 있는 세션은 코드 동기화에 영향을 받지 않습니다 (커밋되지 않은 변경이 있으면 제출을 거부).
- 동시에 도는 arm은 `--slots`개(기본 3)입니다. 이미 시작했거나 큐에 있는 arm은 건너뜁니다.
- 클러스터 설정: `RAX_SLURM_HOST`(Slurm 로그인 노드의 ssh 별칭, 필수), `RAX_REMOTE_ROOT`.
  잡 스크립트는 `exp/slurm/`에 있습니다. 파티션(`all`)과 QoS(`background`)는 저희 클러스터 기준이니 맞게 고치세요.
  `launch.py`는 로그를 `~/slurm-logs/`에 쓰고, 직접 `sbatch` 하는 스크립트는 제출한 디렉터리의 `slurm-logs/`에 씁니다
  (미리 만들어 두세요).

## 5. 대시보드

```bash
bash exp/dashboard/sync.sh            # 클러스터에서 수집(짧은 Slurm 잡) → 복사 → http://localhost:8777
# 로컬 세션이면:
python exp/dashboard/collect.py --sessions runtime/auto-research --out exp/dashboard/site/data
cd exp/dashboard/site && python3 -m http.server 8777
```

- **전체 화면**: phase마다 task × (입력·데모) 칸에 `성공/평가 수`, 그리고 모든 arm의 성공 수,
  에피소드당 시간, 에피소드당 입력·출력 토큰, 총 토큰.
- **arm 화면**: 성공률·시간·토큰 요약, 그리고
  - AgP: **평가 에피소드마다 카드** (성공/실패, 스텝, 분, 출력·입력 토큰, 모델 요청 수, 받은 RGB/Depth/카메라
    파라미터 수, 데모 접근 수, 에이전트의 직접 호출과 코드가 한 호출, 마지막 보고).
  - CaP: 첫 리허설 통과와 제출까지 걸린 시간·출력 토큰, 탐색 대화, 공식 평가 결과, 제출한 `controller.py`.
- 카드를 누르면 그 대화의 **타임라인**이 열립니다: 생각, 응답, 도구 호출, 도구 결과(모델이 실제로 본 이미지 포함),
  이동 결과(도달/막힘, 오차, 스텝), 데모 파일 접근. 필터로 종류별로 볼 수 있습니다.
- 토큰: 입력 = 새 입력 + 캐시 읽기 + 캐시 쓰기 (대부분 캐시 읽기). 사용 한도에 걸린 에피소드는 제외하고 셉니다.

## 6. 코드 구조와 RAX에서 바뀐 점

```
exp/
  arms.yaml            실험 조건 (phase → task × 반복 × arm)
  launch.py            Slurm 제출 (코드 동결, 동시 실행 제한, 중복 방지)
  run_arm.sh           arm 하나 실행 (설정 → 에이전트 실행 → 기록)
  tally.py             arm별 결과 요약
  dashboard/           collect.py (세션 → 대시보드 데이터), site/ (정적 웹), sync.sh
  checks/              실제 시뮬레이터 점검 (입력 프로파일, AgP 이동 도구)
  demos/               데모 요약 keyposes.json (keyposes.py: 공통 요약, robot_keyposes.py: 로봇 데모 재생·요약)
  human_demos/         사람 데모 만들기 (EgoDex 후보 추출, 3D 손 궤적·접촉 단계·GeoRT 손 모양)
  slurm/               양재 잡 스크립트 (설치, 데모, 세션, 점검)
services/  harness/  agent_skills/  auto_research_agent/   RAX 본체 (아래처럼 고침)
```

RAX에서 고친 것:

- **AgP 모드** (`--agent-policy on`): 탐색 에이전트 없이 평가 에피소드마다 아무것도 물려받지 않은 새 에이전트,
  전용 지침과 skill, 목표 단위 이동 도구(`dexjoco_move_ee`, `dexjoco_hand`). CaP는 RAX 그대로.
- **입력 프로파일** `rgb-depth`, `rgb-camera` 추가 (원래 `rgbd`, `rgb-only`).
- **데모 종류** `video`, `human-video`, `human-full` 추가.
- **데모 요약** `keyposes.json` (`exp/demos/`): 데모를 접근 → 접촉 직전 → 접촉 → 쥐기 → 들기 → 놓기 같은 핵심
  순간으로 줄이고, 순간마다 손목 자세(`move_ee` 좌표계), 손 관절(`hand` 순서), 물체 기준 손목 위치를 적습니다.
  로봇 데모는 `setup_gpu.sbatch`가 시뮬레이터에서 재생해 만들고(`KEYPOSES_ONLY=1`이면 요약만 다시),
  사람 데모도 같은 함수(`exp/demos/keyposes.py`의 `summarize`)로 만들 수 있게 했습니다 (물체 위치 칸만 없음).
- 무인 실행 안내, 구독(`claude-login`)에서도 모든 요청을 세션 모델로 고정.
- 로컬 테스트: `runtime/envs/host/bin/python -m pytest -m "not gpu" -q local_tests/unit`
  (새 기능: `test_dexjoco_target_motion.py`, `test_exp_observation_and_experts.py`).
