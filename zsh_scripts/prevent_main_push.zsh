#!/bin/zsh

# Get current branch
prevent_main_push() {
    branch=$(git symbolic-ref --quiet --short HEAD 2>/dev/null)
    echo "[gitpush] Current branch: $branch"

    # Get default branch (primary branch)
    default_branch=$(git remote show origin | awk '/HEAD branch/ {print $NF}')
    echo "[gitpush] Default branch: $default_branch"

    # Block push if on the primary branch
    if [[ "$branch" == "$default_branch" ]]; then
        echo "🚫 Push to '$default_branch' (primary branch) is disabled. Use a pull request instead."
        exit 1
    fi
}
