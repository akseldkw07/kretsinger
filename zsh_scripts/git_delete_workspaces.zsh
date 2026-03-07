#!/bin/zsh

rmworktrees() {
    local REPO_ROOT
    local PRIMARY_BRANCH
    local -a WORKTREE_BRANCHES
    local GIT

    GIT=$(command -v git 2>/dev/null) || {
        echo "[rmworktrees] Error: git not found in PATH."
        return 1
    }

    REPO_ROOT=$("$GIT" rev-parse --show-toplevel 2>/dev/null) || {
        echo "[rmworktrees] Error: Not inside a git repository."
        return 1
    }

    PRIMARY_BRANCH=$("$GIT" symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null)
    if [[ -z "$PRIMARY_BRANCH" ]]; then
        if "$GIT" rev-parse --verify main &>/dev/null; then
            PRIMARY_BRANCH="main"
        elif "$GIT" rev-parse --verify master &>/dev/null; then
            PRIMARY_BRANCH="master"
        else
            echo "[rmworktrees] Error: Could not determine primary branch."
            return 1
        fi
    else
        PRIMARY_BRANCH="${PRIMARY_BRANCH##*/}"
    fi

    WORKTREE_BRANCHES=()

    while read -r wt_path branchref; do
        [[ -z "$wt_path" ]] && continue
        [[ "$wt_path" == "$REPO_ROOT" ]] && continue

        local branch="${branchref#refs/heads/}"

        if [[ -d "$wt_path" ]]; then
            echo "[rmworktrees] 🧹 Removing worktree: $wt_path"
            "$GIT" worktree remove --force "$wt_path" || {
                echo "[rmworktrees] ❌ Failed to remove worktree: $wt_path"
                return 1
            }
        else
            echo "[rmworktrees] 🧹 Worktree directory gone (prunable), skipping remove: $wt_path"
        fi

        if [[ -n "$branch" && "$branch" != "$PRIMARY_BRANCH" ]]; then
            WORKTREE_BRANCHES+=("$branch")
        fi
    done < <(
        "$GIT" worktree list --porcelain |
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

    "$GIT" worktree prune

    if (( ${#WORKTREE_BRANCHES[@]} )); then
        echo "[rmworktrees] 🗑️ Deleting branches attached to removed worktrees..."
        for branch in "${WORKTREE_BRANCHES[@]}"; do
            echo "[rmworktrees] 🗑️ Deleting branch: $branch"
            "$GIT" branch -D "$branch" || {
                echo "[rmworktrees] ❌ Failed to delete branch: $branch"
                return 1
            }
        done
    fi
}
