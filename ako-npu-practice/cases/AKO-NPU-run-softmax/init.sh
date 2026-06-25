#!/bin/bash
# AKO-NPU Environment Initializer
# Sets up Claude Code auto-discovery for NPU skills, agents, and workflows.
#
# Usage:
#   bash init.sh <path-to-skills-repo>
#
# Example:
#   bash init.sh ../skills          # skills repo is at ../skills
#   bash init.sh /path/to/skills    # absolute path
#
# What it does (mirrors CANNBot init.sh for Claude Code):
#   1. Links each skill → .claude/skills/<name>/
#   2. Links each agent → .claude/agents/<name>/
#   3. Links TASK.md → .claude/CLAUDE.md (main entry point)
#   4. Links workflows → .claude/workflows/
#   5. Links settings.local.json for permissions
#   6. Health check

set -e

# --- Color helpers ---
if [ -t 1 ]; then
  GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'
  CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'
else
  GREEN=''; YELLOW=''; RED=''; CYAN=''; BOLD=''; DIM=''; NC=''
fi

ok()   { echo -e "  ${DIM}${GREEN}✓${NC}${DIM} $*${NC}"; }
warn() { echo -e "  ${YELLOW}⚠${NC}${DIM} $*${NC}"; }
err()  { echo -e "  ${RED}✗${NC}${DIM} $*${NC}"; }
step() { echo -e "${DIM}$*${NC}"; }

# --- Parse args ---
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    cat << 'EOF'
AKO-NPU Environment Initializer

Usage: bash init.sh <path-to-skills-repo>

Arguments:
  path-to-skills-repo  Path to the CANN skills repository (contains skills/, agents/, teams/)

Example:
  bash init.sh ../skills
  bash init.sh /home/user/kernel/autoresearch/skills

What it does:
  - Links NPU skills into .claude/skills/ (Claude Code auto-discovers them)
  - Links NPU agents into .claude/agents/ (Claude Code auto-discovers them)
  - Links TASK.md as .claude/CLAUDE.md (Claude Code reads at startup)
  - Links workflows into .claude/workflows/
  - Sets up settings.local.json with permissions
EOF
    exit 0
fi

SKILLS_REPO="${1:-}"
if [ -z "$SKILLS_REPO" ]; then
    err "Missing argument: path to skills repo"
    echo "  Usage: bash init.sh <path-to-skills-repo>"
    echo "  Example: bash init.sh ../skills"
    exit 1
fi

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "$SKILLS_REPO" ]; then
    err "Skills repo not found: $SKILLS_REPO"
    exit 1
fi
SKILLS_REPO="$(cd "$SKILLS_REPO" && pwd)"

# Verify it's actually a skills repo
if [ ! -d "$SKILLS_REPO/skills" ]; then
    err "Invalid skills repo: $SKILLS_REPO/skills/ not found"
    exit 1
fi

CONFIG_ROOT="$SCRIPT_DIR/.claude"

echo ""
echo -e "  ${BOLD}AKO-NPU Init${NC}"
echo ""
echo "  Project:     $SCRIPT_DIR"
echo "  Skills repo: $SKILLS_REPO"
echo "  Config:      $CONFIG_ROOT"
echo ""

# --- Step 1: Skills ---
step "[1/5] Linking skills..."
mkdir -p "$CONFIG_ROOT/skills"

# Clean existing skill symlinks
for link in "$CONFIG_ROOT/skills"/*/; do
    link="${link%/}"
    [ -L "$link" ] && rm "$link"
done

skill_count=0
for skill_dir in "$SKILLS_REPO/skills"/*/; do
    [ -d "$skill_dir" ] || continue
    name=$(basename "$skill_dir")
    ln -sfn "$(realpath "$skill_dir")" "$CONFIG_ROOT/skills/$name"
    skill_count=$((skill_count + 1))
done
ok "Skills: $skill_count linked"
echo ""

# --- Step 2: Agents ---
step "[2/5] Linking agents..."
mkdir -p "$CONFIG_ROOT/agents"

# Clean existing agent symlinks
for link in "$CONFIG_ROOT/agents"/*/; do
    link="${link%/}"
    [ -L "$link" ] && rm "$link"
done

agent_count=0
# Link ascendc-kernel-* agents (architect, developer, reviewer)
for agent_dir in "$SKILLS_REPO/agents"/*/; do
    [ -d "$agent_dir" ] || continue
    name=$(basename "$agent_dir")
    ln -sfn "$(realpath "$agent_dir")" "$CONFIG_ROOT/agents/$name"
    agent_count=$((agent_count + 1))
done
ok "Agents: $agent_count linked"
echo ""

# --- Step 3: CLAUDE.md + Workflows ---
step "[3/5] Linking CLAUDE.md and workflows..."

# TASK.md → .claude/CLAUDE.md (Claude Code reads this at startup)
ln -sf "$(realpath "$SCRIPT_DIR/TASK.md")" "$CONFIG_ROOT/CLAUDE.md"
ok "CLAUDE.md → TASK.md"

# Link workflows from the team config
TEAM_DIR="$SKILLS_REPO/teams/ops-direct-invoke"
if [ -d "$TEAM_DIR/workflows" ]; then
    ln -sfn "$(realpath "$TEAM_DIR/workflows")" "$CONFIG_ROOT/workflows"
    ok "workflows/ linked"
else
    warn "workflows/ not found in $TEAM_DIR"
fi
echo ""

# --- Step 4: Settings ---
step "[4/5] Configuring settings..."

# Preserve existing settings.local.json or create default
if [ ! -f "$CONFIG_ROOT/settings.local.json" ]; then
    cat > "$CONFIG_ROOT/settings.local.json" << 'SETTINGS_EOF'
{
  "permissions": {
    "allow": [
      "Bash(*)",
      "Read(*)",
      "Write(*)",
      "Edit(*)",
      "Glob(*)",
      "Grep(*)",
      "Agent(*)",
      "WebFetch(*)",
      "WebSearch(*)"
    ]
  }
}
SETTINGS_EOF
    ok "settings.local.json created"
else
    ok "settings.local.json exists (preserved)"
fi
echo ""

# --- Step 5: Health check ---
step "[5/5] Health check..."
health_ok=true

# Check skills
skill_actual=$(ls -d "$CONFIG_ROOT/skills"/*/ 2>/dev/null | wc -l)
if [ "$skill_actual" -gt 0 ]; then
    ok "Skills: $skill_actual available"
else
    err "Skills: none found"
    health_ok=false
fi

# Check agents
agent_actual=$(ls -d "$CONFIG_ROOT/agents"/*/ 2>/dev/null | wc -l)
if [ "$agent_actual" -gt 0 ]; then
    ok "Agents: $agent_actual available"
else
    warn "Agents: none found"
fi

# Check CLAUDE.md
if [ -f "$CONFIG_ROOT/CLAUDE.md" ]; then
    ok "CLAUDE.md present"
else
    err "CLAUDE.md missing"
    health_ok=false
fi

# Check workflows
if [ -d "$CONFIG_ROOT/workflows" ]; then
    ok "workflows/ present"
else
    warn "workflows/ missing"
fi

echo ""
if [ "$health_ok" = true ]; then
    echo -e "  ${GREEN}${BOLD}✓ AKO-NPU initialized successfully!${NC}"
else
    echo -e "  ${RED}${BOLD}✗ Some checks failed, see above${NC}"
fi

echo ""
echo -e "  ${BOLD}Quick Start:${NC}"
echo -e "  ${CYAN}1.${NC} Place kernel files in ${GREEN}input/${NC}"
echo -e "  ${CYAN}2.${NC} Run: ${GREEN}cd $(basename "$SCRIPT_DIR") && claude${NC}"
echo -e "  ${CYAN}3.${NC} Say: ${GREEN}${BOLD}Follow the instructions in TASK.md${NC}"
echo ""
