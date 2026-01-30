#!/bin/zsh
# checkout main, delete branches, and pull latest changes

mpgd() {
    local PRIMARY_BRANCH

    # Determine primary branch
    PRIMARY_BRANCH=$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null)
    if [[ -z "$PRIMARY_BRANCH" ]]; then
        if git rev-parse --verify main &>/dev/null; then
            PRIMARY_BRANCH="main"
        elif git rev-parse --verify master &>/dev/null; then
            PRIMARY_BRANCH="master"
        else
            echo "Error: Could not determine primary branch."
            return 1
        fi
    else
        PRIMARY_BRANCH="${PRIMARY_BRANCH##*/}"
    fi

    echo "[mpgd] ✅ Primary branch is: $PRIMARY_BRANCH"

    echo "[mpgd] 🧹 Removing all worktrees (except the primary worktree) and associated branches..."

    local REPO_ROOT
    local -a WORKTREE_BRANCHES=()

    REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
        echo "Error: Not inside a git repository."
        return 1
    }

    while read -r path branchref; do
        [[ -z "$path" ]] && continue
        [[ "$path" == "$REPO_ROOT" ]] && continue

        local branch="${branchref#refs/heads/}"

        echo "[mpgd] 🧹 Removing worktree: $path"
        git worktree remove --force "$path"

        if [[ -n "$branch" ]]; then
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

    if (( ${#WORKTREE_BRANCHES[@]} )); then
        echo "[mpgd] 🗑️ Deleting branches attached to removed worktrees..."
        for branch in "${WORKTREE_BRANCHES[@]}"; do
            if [[ "$branch" != "$PRIMARY_BRANCH" ]]; then
                echo "[mpgd] 🗑️ Deleting branch: $branch"
                git branch -D "$branch"
            else
                echo "[mpgd] ⚠️ Skipping primary branch: $branch"
            fi
        done
    fi

    git checkout "$PRIMARY_BRANCH" && git pull origin "$PRIMARY_BRANCH"

    echo "[mpgd] 🔍 Deleting all local branches except '$PRIMARY_BRANCH'..."

    git for-each-ref --format='%(refname:short)' refs/heads/ | while read -r branch; do
        if [[ "$branch" != "$PRIMARY_BRANCH" ]]; then
            echo "[mpgd] 🗑️ Deleting branch: $branch"
            git branch -D "$branch"
        fi
    done
}
