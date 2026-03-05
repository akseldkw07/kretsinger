#!/bin/zsh
# Diagnostic version of mpgd

mpgd_debug() {
    echo "=== PATH at start of function ==="
    echo $PATH | tr ':' '\n' | head -10
    echo ""

    echo "=== Which git ==="
    which git
    echo ""

    echo "=== Running git worktree list ==="
    git worktree list
    echo ""

    echo "=== PATH before process substitution ==="
    echo $PATH | tr ':' '\n' | head -10
    echo ""

    local REPO_ROOT
    REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)

    while read -r path branchref; do
        echo "=== Inside while loop ==="
        echo "PATH: $(echo $PATH | tr ':' '\n' | head -5)"
        echo "Which git: $(which git)"
        echo "Worktree path: $path"
        break  # Just check first iteration
    done < <(git worktree list --porcelain | awk '/^worktree / { path=$2 } /^branch / { branch=$2 } /^$/ { if (path != "") print path, branch; path=""; branch="" } END { if (path != "") print path, branch }')
}
