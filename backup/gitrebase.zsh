#!/bin/zsh

rebase_squash_conflict() {
    local feature_branch
    feature_branch=$(git rev-parse --abbrev-ref HEAD)

    echo "[RebaseSquash] Saving current changes on '$feature_branch'..."
    git add -A
    git commit -m "WIP (pre-squash rebase)"

    echo "[RebaseSquash] Checking out main and updating..."
    git checkout main || return 1
    git pull origin main || return 1

    echo "[RebaseSquash] Returning to '$feature_branch'..."
    git checkout "$feature_branch" || return 1

    echo "[RebaseSquash] Squashing feature branch into a single commit..."
    git reset --soft main
    git commit -m "Squashed commit from feature branch"

    echo "[RebaseSquash] Rebasing onto updated main (manual conflict resolution)..."
    git rebase main

    echo "[RebaseSquash] Force-pushing changes..."
    git push --force
}

rebase_nuclear_feature() {
    local feature_branch
    feature_branch=$(git rev-parse --abbrev-ref HEAD)

    echo "[rebase_nuclear_feature] 💥 WARNING: You are about to rebase '$feature_branch' onto 'main'"
    echo "[rebase_nuclear_feature] ⚠️  Conflicts will be resolved in favor of the feature branch—main's changes may be discarded."
    echo "[rebase_nuclear_feature] 🚧 This operation rewrites history and will force-push to remote."
    read -q "?Do you want to continue with the nuclear rebase? (y/n) " && echo
    if [[ $REPLY != [Yy] ]]; then
        echo "[rebase_nuclear_feature] ❌ Rebase aborted by user."
        return 1
    fi

    echo "[rebase_nuclear_feature] 🛠️  Committing current changes on '$feature_branch'..."
    git add -A
    git commit -m "WIP (pre-nuclear rebase)"

    echo "[rebase_nuclear_feature] ⬇️  Updating main..."
    git checkout main && git pull origin main || return 1

    echo "[rebase_nuclear_feature] 🔁 Switching back to '$feature_branch'..."
    git checkout "$feature_branch" || return 1

    echo "[rebase_nuclear_feature] 🧨 Rebasing with 'theirs' strategy (feature branch overrides main)..."
    git rebase -s recursive -X theirs main || return 1

    echo "[rebase_nuclear_feature] 🚀 Force-pushing updated feature branch..."
    git push --force
}
