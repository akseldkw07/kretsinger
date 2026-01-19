#!/bin/zsh
# checkout main, delete branches, and pull latest changes
# TODO delete all worktrees, and then delete corresponding branches
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

    git checkout "$PRIMARY_BRANCH" && git pull origin "$PRIMARY_BRANCH"

    echo "[mpgd] 🔍 Deleting all local branches except '$PRIMARY_BRANCH'..."

    git for-each-ref --format='%(refname:short)' refs/heads/ | while read -r branch; do
        if [[ "$branch" != "$PRIMARY_BRANCH" ]]; then
            echo "[mpgd] 🗑️ Deleting branch: $branch"
            git branch -D "$branch"
        fi
    done
}
