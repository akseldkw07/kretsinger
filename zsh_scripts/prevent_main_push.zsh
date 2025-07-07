#!/bin/zsh

prevent_main_push() {
    branch=$(git symbolic-ref --quiet --short HEAD 2>/dev/null)
    echo "[gitpush] Current branch: $branch"

    # Get default branch from local config (fallback to 'main' if not set)
    default_branch=$(git config --local --get init.defaultBranch)
    if [[ -z "$default_branch" ]]; then
        default_branch=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@')
    fi
    if [[ -z "$default_branch" ]]; then
        default_branch="main"
    fi
    echo "[gitpush] Default branch: $default_branch"

    if [[ "$branch" == "$default_branch" ]]; then
        echo "🚫 Push to '$default_branch' (primary branch) is disabled. Use a pull request instead."
        exit 1
    fi
}
