#!/usr/bin/env bash

rebase_squash_conflict() {
  local feature_branch
  feature_branch="$(git rev-parse --abbrev-ref HEAD)"

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
  local feature_branch stash_was_created=false
  feature_branch="$(git rev-parse --abbrev-ref HEAD)"

  if [[ "$feature_branch" == "main" ]]; then
    echo "[nuke] ❌ You’re on 'main'. Run this from a feature branch."
    return 1
  fi

  echo "[nuke] 💥 You’re about to rebase '$feature_branch' onto 'main', prioritizing YOUR changes."
  echo "[nuke] ⚠️ This overwrites history and will force-push to remote."
  local reply=""
  read -r -n 1 -p "Proceed with nuclear rebase? (y/n) " reply
  echo
  if [[ ! "$reply" =~ ^[Yy]$ ]]; then
    echo "[nuke] 🛑 Rebase aborted by user."
    return 1
  fi

  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "[nuke] 💾 Stashing uncommitted changes..."
    git stash push -u -m "pre-nuclear-rebase"
    stash_was_created=true
  fi

  echo "[nuke] 🧷 Committing current staged work..."
  git add -A
  git commit -m "WIP (pre-nuclear rebase)"

  echo "[nuke] 🔄 Switching to 'main'..."
  if ! git checkout main || ! git pull origin main; then
    echo "[nuke] ❌ Failed to update 'main'. Returning to '$feature_branch'..."
    git checkout "$feature_branch"
    return 1
  fi

  echo "[nuke] 🔁 Switching back to '$feature_branch'..."
  git checkout "$feature_branch" || return 1

  echo "[nuke] 🧨 Rebasing with 'theirs' strategy (your branch wins over main)..."
  if ! git rebase -s recursive -X theirs main; then
    echo "[nuke] ⚠️ Rebase failed. Resolve conflicts manually and run 'git rebase --continue'."
    return 1
  fi

  echo "[nuke] 🚀 Force-pushing rebased '$feature_branch'..."
  git push --force

  if [[ "$stash_was_created" == true ]]; then
    echo "[nuke] 🌱 Re-applying stashed changes..."
    git stash pop
  fi

  echo "[nuke] ✅ Rebase complete."
}
