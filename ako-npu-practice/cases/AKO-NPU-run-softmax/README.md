<h1 align="center">AKO-NPU</h1>
<p align="center"><b>Agentic Kernel Optimization for NPU</b></p>

Automated Ascend C kernel optimization on Huawei NPU, powered by coding agents with externalized NPU skills. Provide any Ascend C kernel — the agent iteratively rewrites it for maximum performance by consulting NPU-specific skills at every decision point.

> **Relationship to AKO4ALL:** AKO-NPU adapts [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL)'s agentic iteration methodology from GPU to NPU. The key difference: AKO4ALL relies on model-internalized GPU knowledge (CUDA/Triton), while AKO-NPU externalizes NPU knowledge into skills — the agent consults these skills rather than relying on built-in knowledge.

## Quick Start

### 1. Initialize

```bash
cd AKO-NPU

# Initialize with path to CANN skills repo
bash init.sh ../skills
```

`init.sh` sets up Claude Code auto-discovery:
- Links NPU skills → `.claude/skills/` (auto-discovered by Claude Code)
- Links NPU agents → `.claude/agents/` (auto-discovered by Claude Code)
- Links TASK.md → `.claude/CLAUDE.md` (read at startup)
- Links workflows → `.claude/workflows/`

### 2. Place kernel files

```
AKO-NPU/
├── input/                       # Place your kernel files here
│   ├── kernel.asc               # Ascend C kernel source
│   ├── CMakeLists.txt           # Build config
│   └── ...
├── context/                     # Optional reference materials
```

### 3. Run

```bash
claude
```

Then say: `Follow the instructions in TASK.md`

Or with bypass permissions: `claude --dangerously-skip-permissions`

## What Happens

1. **Setup** — The agent uses `ascendc-env-check` to verify the environment, creates an optimization branch, copies the kernel to `solution/`, builds a benchmark script using `ascendc-direct-invoke-template` and `ops-profiling`, and verifies correctness.
2. **Profile** — Uses `ops-profiling` to run msprof on the baseline, identifying bottlenecks via CSV analysis.
3. **Iterate** — Each iteration: modify kernel → benchmark → log to `ITERATIONS.md` → git commit. The agent consults `ascendc-tiling-design`, `ascendc-api-best-practices`, `ascendc-npu-arch` for optimization strategies.
4. **Track** — Every iteration is saved to `trajectory/` with the kernel source and benchmark output.

## AKO4ALL vs AKO-NPU

| Aspect | AKO4ALL | AKO-NPU |
|--------|---------|---------|
| Knowledge source | Model-internalized + Web Search | External skills (auto-discovered by Claude Code) |
| bench.py | Hardcoded GPU evaluation script | None — agent builds bench from skills |
| Profiling | Hardcoded `ncu` commands | None — agent uses `ops-profiling` skill |
| Skills | None | Auto-discovered via `.claude/skills/` |
| Agents | None | `ascendc-kernel-{architect,developer,reviewer}` |
| Fixed scripts | bench.py, bench-wrapper.sh | Only bench-wrapper.sh (trajectory shell) |

## Project Structure

```
AKO-NPU/
├── init.sh              # Environment initializer (run first)
├── TASK.md              # Optimization main loop rules (→ .claude/CLAUDE.md)
├── HINTS.md             # Behavior control + skills usage guidance
├── ITERATIONS.md        # Iteration log template
├── bench-wrapper.sh     # Trajectory tracking shell (template)
├── input/               # User places kernel files here
├── context/             # Optional reference materials
├── README.md
├── .gitignore
└── .claude/
    ├── settings.local.json   # Permissions (committed)
    ├── skills/               # → NPU skills (init.sh generates)
    ├── agents/               # → NPU agents (init.sh generates)
    ├── workflows/            # → Workflow definitions (init.sh generates)
    └── CLAUDE.md             # → TASK.md (init.sh generates)
```

## Available Skills (after init)

| Skill | Purpose |
|-------|---------|
| `ascendc-env-check` | NPU environment verification |
| `ascendc-direct-invoke-template` | Build/run project template |
| `ascendc-api-best-practices` | Ascend C API usage patterns |
| `ascendc-tiling-design` | Tiling strategy methodology |
| `ascendc-npu-arch` | NPU hardware specs |
| `ops-profiling` | msprof profiling and analysis |
| `ascendc-precision-debug` | Accuracy debugging |
| `ascendc-runtime-debug` | Runtime error diagnosis |
| `ascendc-code-review` | Code quality review |
| `ascendc-docs-search` | API documentation search |

## Available Agents (after init)

| Agent | Role |
|-------|------|
| `ascendc-kernel-architect` | Architecture design, tiling strategy |
| `ascendc-kernel-developer` | Code implementation, build & test |
| `ascendc-kernel-reviewer` | Code review, quality scoring |

## Hints

`HINTS.md` controls agent behavior. You can add directives such as:

- **Optimization constraints** — focus areas, techniques to avoid
- **Strategies** — specific approaches to try or skip
- **Agent behavior** — iteration limits, verbosity
- **Dependency policies** — whether the agent may install packages

## Acknowledgments

- [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL) — The methodology and iteration protocol are adapted from AKO4ALL.
- [CANN Skills](https://gitcode.com/cann/skills) — NPU development skills that power the knowledge layer.
