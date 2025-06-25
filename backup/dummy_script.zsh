#!/bin/zsh
PRIMARY_BRANCH=$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null)
if [ -z "$PRIMARY_BRANCH" ]; then
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

echo $PRIMARY_BRANCH
git checkout "${PRIMARY_BRANCH}"
git pull origin "${PRIMARY_BRANCH}"
