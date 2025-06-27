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
    local current_branch feature_branch stash_was_created=false
    feature_branch=$(git rev-parse --abbrev-ref HEAD)

    if [[ "$feature_branch" == "main" ]]; then
        echo "[nuke] ❌ You’re on 'main'. This function must be run from a feature branch."
        return 1
    fi

    echo "[nuke] 💥 You’re about to rebase '$feature_branch' onto 'main', prioritizing YOUR changes."
    echo "[nuke] ⚠️ This overwrites history and will force-push to remote."
    read -q "?Do you want to proceed with the nuclear rebase? (y/n) " && echo
    if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
        echo "[nuke] 🛑 Rebase aborted by user."
        return 1
    fi

    # 🔒 Stash if needed
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "[nuke] 💾 Stashing uncommitted changes..."
        git stash push -u -m "pre-nuclear-rebase"
        stash_was_created=true
    fi

    # 💾 Save WIP commit
    echo "[nuke] 🧷 Committing current staged work..."
    git add -A
    git commit -m "WIP (pre-nuclear rebase)"

    # 🔄 Update main
    echo "[nuke] 🔄 Switching to 'main'..."
    if ! git checkout main || ! git pull origin main; then
        echo "[nuke] ❌ Failed to update 'main'. Returning to '$feature_branch'..."
        git checkout "$feature_branch"
        return 1
    fi

    # 🔁 Return to feature branch
    echo "[nuke] 🔁 Switching back to '$feature_branch'..."
    if ! git checkout "$feature_branch"; then
        echo "[nuke] ❌ Failed to return to feature branch."
        return 1
    fi

    # 🧨 Perform nuclear rebase
    echo "[nuke] 🧨 Rebasing with 'theirs' strategy (your branch wins over main)..."
    if ! git rebase -s recursive -X theirs main; then
        echo "[nuke] ⚠️ Rebase failed. Resolve conflicts manually and re-run 'git rebase --continue'."
        return 1
    fi

    # 🚀 Push
    echo "[nuke] 🚀 Force-pushing rebased '$feature_branch'..."
    git push --force

    # 🌱 Restore stash if one was created
    if [[ "$stash_was_created" == true ]]; then
        echo "[nuke] 🌱 Re-applying stashed changes..."
        git stash pop
    fi

    echo "[nuke] ✅ Rebase complete. '$feature_branch' is now aligned (and dominant) over 'main'."
}
