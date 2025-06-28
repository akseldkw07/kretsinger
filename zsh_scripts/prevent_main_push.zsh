#!/bin/zsh

# Get current branch
branch=$(git symbolic-ref --quiet --short HEAD 2>/dev/null)

# Block push if on main
if [[ "$branch" == "main" ]]; then
    echo "🚫 Push to 'main' is disabled. Use a pull request instead."
    exit 1
fi
