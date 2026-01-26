#!/usr/bin/env bash
# pre-commit hook to prevent pushing to the main branch directly

prevent_main_push() {
  local branch default_branch
  branch="$(git symbolic-ref --quiet --short HEAD 2>/dev/null)"
  echo "[gitpush] Current branch: $branch"

  default_branch="$(git config --local --get init.defaultBranch)"
  if [[ -z "$default_branch" ]]; then
    default_branch="$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@')"
  fi
  [[ -z "$default_branch" ]] && default_branch="main"
  echo "[gitpush] Default branch: $default_branch"

  if [[ "$branch" == "$default_branch" ]]; then
    echo "🚫 Push to '$default_branch' (primary branch) is disabled. Use a pull request instead."
    exit 1
  fi
}
