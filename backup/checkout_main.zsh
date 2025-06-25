#!/bin/zsh
mpgd() {
    PRIMARY_BRANCH=$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null)
    if [ -z "$PRIMARY_BRANCH" ]; then
        if git rev-parse --verify main &>/dev/null; then
            PRIMARY_BRANCH="main"
        elif git rev-parse --verify master &>/dev-null; then
            PRIMARY_BRANCH="master"
        else
            echo "Error: Could not determine primary branch."
            return 1
        fi
    else
        PRIMARY_BRANCH="${PRIMARY_BRANCH##*/}"
    fi
    git checkout "${PRIMARY_BRANCH}"
    git pull origin "${PRIMARY_BRANCH}"

    # Robustly iterate over branches using 'while read' loop to prevent word splitting issues
    git branch | grep -v "^*" | grep -v "${PRIMARY_BRANCH}" | sed "s/^[[:space:]]*//p" | while IFS= read -r branch; do
        if [ -n "$branch" ]; then # Ensure the branch name is not empty
            echo "Deleting branch: $branch"
            git branch -D "${branch}" # Force delete branch
        fi
    done
}
