#!/bin/zsh

gitpush() {
  local commit_message="$1"
  git add .

  # 🔍 Collect diff summary
  local diff_output
  diff_output=$(git diff --cached --name-status)

  local added=()
  local modified=()
  local deleted=()

  while IFS=$'\t' read -r change_type file; do
    filename="${file##*/}"
    case "$change_type" in
    A) added+=("$filename") ;;
    M) modified+=("$filename") ;;
    D) deleted+=("$filename") ;;
    esac
  done <<<"$diff_output"

  # Build a nicely formatted summary for GitHub UI
  summary=""
  if ((${#added[@]} > 0)); then
    summary+=$'\nAdded:'
    for file in "${added[@]}"; do
      summary+=$'\n- '"$file"
    done
  fi
  if ((${#modified[@]} > 0)); then
    summary+=$'\nModified:'
    for file in "${modified[@]}"; do
      summary+=$'\n- '"$file"
    done
  fi
  if ((${#deleted[@]} > 0)); then
    summary+=$'\nDeleted:'
    for file in "${deleted[@]}"; do
      summary+=$'\n- '"$file"
    done
  fi

  if [[ -z "$commit_message" ]]; then
    commit_message=$(date +"%Y-%m-%d %H:%M:%S")
  fi

  local full_message="$commit_message"
  [[ -n "$summary" ]] && full_message+="$summary"

  # 🖨️ Show summary
  echo
  print -P "%F{cyan}${commit_message}%f"
  ((${#added[@]} > 0)) && print -P "%F{green}add:%f ${added[*]}"
  ((${#modified[@]} > 0)) && print -P "%F{yellow}modify:%f ${modified[*]}"
  ((${#deleted[@]} > 0)) && print -P "%F{red}delete:%f ${deleted[*]}"
  echo

  # ✅ Commit and only push if commit succeeds
  if git commit -m "$full_message"; then
    local push_output pr_url
    # Check if gh CLI is installed
    if command -v gh >/dev/null 2>&1; then
      # Use gh to push and open PR if needed
      push_output=$(git push --force 2>&1 | tee /dev/tty)
      if echo "$push_output" | grep -q "Create a pull request for"; then
        pr_url=$(echo "$push_output" | grep -Eo 'https://github\\.com/[^ ]+')
        if [[ -n "$pr_url" ]]; then
          echo "[gitpush] 🚀 Opening pull request in browser..."
          open "$pr_url"
        fi
      else
        echo "[gitpush] ✅ Push complete. Pull request already exists."
      fi
    else
      # Fallback: just push
      if git push --force; then
        echo "[gitpush] ✅ Push complete."
      else
        echo "[gitpush] ❌ Push failed."
        return 1
      fi
    fi
  else
    echo "[gitpush] ❌ Commit failed. Push aborted."
    return 1
  fi
}
