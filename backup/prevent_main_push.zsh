#!/bin/zsh

# Get current branch
prevent_main_push() {
    branch=$(git symbolic-ref --quiet --short HEAD 2>/dev/null)

    # Get default branch (primary branch)
    default_branch=$(git symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null | sed 's@^origin/@@')

    # Block push if on the primary branch
    if [[ "$branch" == "$default_branch" ]]; then
        echo "🚫 Push to '$default_branch' (primary branch) is disabled. Use a pull request instead."
        exit 1
    fi
}
