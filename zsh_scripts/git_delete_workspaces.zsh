#!/bin/zsh

rmworkspaces() {
    local REPO_ROOT
    local PRIMARY_BRANCH
    local -a WORKTREE_BRANCHES

    REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
        echo "[rmworkspaces] Error: Not inside a git repository."
        return 1
    }

    PRIMARY_BRANCH=$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null)
    if [[ -z "$PRIMARY_BRANCH" ]]; then
        if git rev-parse --verify main &>/dev/null; then
            PRIMARY_BRANCH="main"
        elif git rev-parse --verify master &>/dev/null; then
            PRIMARY_BRANCH="master"
        else
            echo "[rmworkspaces] Error: Could not determine primary branch."
            return 1
        fi
    else
        PRIMARY_BRANCH="${PRIMARY_BRANCH##*/}"
    fi

    WORKTREE_BRANCHES=()

    while read -r path branchref; do
        [[ -z "$path" ]] && continue
        [[ "$path" == "$REPO_ROOT" ]] && continue

        local branch="${branchref#refs/heads/}"

        if [[ -d "$path" ]]; then
            echo "[rmworkspaces] 🧹 Removing worktree: $path"
            git worktree remove --force "$path" || {
                echo "[rmworkspaces] ❌ Failed to remove worktree: $path"
                return 1
            }
        else
            echo "[rmworkspaces] 🧹 Worktree directory gone (prunable), skipping remove: $path"
        fi

        if [[ -n "$branch" && "$branch" != "$PRIMARY_BRANCH" ]]; then
            WORKTREE_BRANCHES+=("$branch")
        fi
    done < <(
        git worktree list --porcelain |
            awk '
                /^worktree / { path=$2 }
                /^branch /   { branch=$2 }
                /^$/ {
                    if (path != "") print path, branch
                    path=""; branch=""
                }
                END {
                    if (path != "") print path, branch
                }
            '
    )

    git worktree prune

    if (( ${#WORKTREE_BRANCHES[@]} )); then
        echo "[rmworkspaces] 🗑️ Deleting branches attached to removed worktrees..."
        for branch in "${WORKTREE_BRANCHES[@]}"; do
            echo "[rmworkspaces] 🗑️ Deleting branch: $branch"
            git branch -D "$branch" || {
                echo "[rmworkspaces] ❌ Failed to delete branch: $branch"
                return 1
            }
        done
    fi
}
