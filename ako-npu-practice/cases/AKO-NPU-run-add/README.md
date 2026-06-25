<h1 align="center">AKO-NPU</h1>
<p align="center"><b>Agentic Kernel Optimization for NPU</b></p>

Automated Ascend C kernel optimization on Huawei NPU, powered by coding agents with externalized NPU skills. Provide any Ascend C kernel — the agent iteratively rewrites it for maximum performance by consulting NPU-specific skills at every decision point.

> **Relationship to AKO4ALL:** AKO-NPU adapts [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL)'s agentic iteration methodology from GPU to NPU. The key difference: AKO4ALL relies on model-internalized GPU knowledge (CUDA/Triton), while AKO-NPU externalizes NPU knowledge into a `skills/` directory — the agent consults these skills rather than relying on built-in knowledge.

## AKO4ALL vs AKO-NPU

| Aspect | AKO4ALL | AKO-NPU |
|--------|---------|---------|
| Knowledge source | Model-internalized + Web Search | External skills (independently updatable) |
| bench.py | Hardcoded GPU evaluation script | None — agent builds bench from skills |
| Profiling | Hardcoded `ncu` commands | None — agent learns profiling from skills |
| Skill references | None | Generic guidance ("consult skills") |
| Fixed scripts | bench.py, bench-wrapper.sh | Only bench-wrapper.sh (trajectory shell) |

```
AKO4ALL (GPU):                          AKO-NPU (NPU):
┌──────────┐   ┌─────────────┐          ┌──────────┐   ┌─────────────┐   ┌──────────┐
│ TASK.md  │──▶│ Claude Code │           │ TASK.md  │──▶│ Claude Code │──▶│NPU Skills│
│ HINTS.md │   │  (Agent)    │           │ HINTS.md │   │  (Agent)    │   │(External)│
└──────────┘   └─────────────┘           └──────────┘   └─────────────┘   └──────────┘
               Model-internalized                        Skills = externalized knowledge
               GPU knowledge                             (can be updated independently)
```

## What You Provide

- **Kernel** (required) — The Ascend C kernel to optimize. Place in `input/`.
- **Context** (optional) — Reference materials: algorithm descriptions, papers, design docs. Place in `context/`.
- **Hints** (optional) — Directives for the agent: constraints, focus areas, behavior controls. Edit `HINTS.md`.
- **Skills** (required) — NPU development skills must be accessible at `skills/` (symlink or actual directory).

## Requirements

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (or compatible coding agent)
- Huawei Ascend NPU environment (CANN toolkit installed, Ascend C compiler available)
- NPU development skills (`skills/` directory)
- Git

> **Note:** No `bench.py` or profiling scripts are hardcoded. The agent reads the skills to learn how to compile, benchmark, and profile on NPU, then builds the evaluation pipeline itself.

## Quick Start

1. Ensure the NPU environment is available (CANN toolkit, Ascend C compiler).

2. Ensure the `skills/` directory is accessible from within `AKO-NPU/`:

```bash
# Option A: symlink (recommended)
cd AKO-NPU && ln -s /path/to/skills skills

# Option B: skills already at ../skills (default layout)
```

3. Place your kernel files:

```
AKO-NPU/
├── input/                       # Place your kernel files here
│   ├── add_custom.cpp           # Example — Ascend C kernel source
│   └── ...
├── context/                     # Place reference materials here (optional)
├── skills/ -> ../skills         # NPU development skills (required)
├── HINTS.md                     # Edit to add directives (optional)
```

4. Run:

```bash
cd AKO-NPU && claude
```

5. Start optimization:

```
Follow the instructions in TASK.md
```

## What Happens

1. **Setup** — The agent reads your files, **consults skills** to understand the NPU environment and compilation process, creates an optimization branch, copies the kernel to `solution/`, builds a benchmark script, and verifies correctness.
2. **Profile** — Consults skills for NPU profiling methods, runs profiling on the baseline kernel to identify bottlenecks.
3. **Iterate** — Each iteration: modify kernel → benchmark → log results to `ITERATIONS.md` → git commit. The agent consults skills for optimization strategies (tiling, pipeline, memory, API selection).
4. **Track** — Every iteration is saved to `trajectory/` with the kernel source and benchmark output.

## Hints

`HINTS.md` controls agent behavior. You can add directives such as:

- **Optimization constraints** — focus areas, techniques to avoid
- **Strategies** — specific approaches to try or skip
- **Agent behavior** — iteration limits, verbosity
- **Dependency policies** — whether the agent may install packages

## Permissions

The optimization loop involves running shell commands (compiling, benchmarking, profiling). To run fully unattended, grant the necessary permissions.

For Claude Code, the simplest option:

```bash
cd AKO-NPU && claude --dangerously-skip-permissions
```

For granular control, the `.claude/settings.local.json` is pre-configured with recommended permissions.

## Skills Dependency

AKO-NPU's core design principle is that NPU knowledge lives in external skills, not hardcoded in rules. The `skills/` directory should contain:

- **Environment setup** — How to check and configure the NPU environment
- **Ascend C API** — Operator development API reference
- **Tiling design** — Methodology for tiling strategies
- **Performance analysis** — Profiling tools and metric interpretation
- **Precision debugging** — Accuracy verification methods
- **NPU architecture** — Hardware characteristics and constraints
- **Project templates** — Build system and project structure patterns

When skills are updated (new APIs, better profiling methods, new optimization patterns), AKO-NPU automatically benefits — no rule changes needed.

## Project Structure

```
AKO-NPU/
├── TASK.md              # Optimization main loop rules (core)
├── HINTS.md             # Behavior control + skills usage guidance
├── ITERATIONS.md        # Iteration log
├── bench-wrapper.sh     # Trajectory tracking shell (template)
├── input/               # User places kernel files here
├── context/             # Optional reference materials
├── README.md
├── .gitignore
└── .claude/
    └── settings.local.json
```

Generated at runtime by the agent:
```
├── solution/            # Working copy of the kernel (agent creates)
├── scripts/
│   └── bench.sh         # Benchmark script (agent creates from bench-wrapper.sh)
└── trajectory/          # Iteration snapshots (auto-generated)
```

## Tips

- **Consult skills first** — If the agent seems confused about NPU concepts, ensure `skills/` is properly linked and contains relevant knowledge.
- **Model matters** — Model capability influences optimization quality. We recommend [Claude Opus 4.6](https://docs.anthropic.com/en/docs/about-claude/models).
- **Iteration limits** — By default, there is no iteration cap. To set one, add to `HINTS.md` or your prompt (e.g., `Optimize for up to 20 iterations.`).
- **Intervene anytime** — You can interrupt to give guidance, discuss strategy, or manually edit files in `solution/`.

## Acknowledgments

- [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL) — The methodology and iteration protocol are adapted from AKO4ALL.
- [autoresearch](https://github.com/karpathy/autoresearch) and [autokernel](https://github.com/RightNow-AI/autokernel) — Inspiration for autonomous optimization loops.
