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

  local added_str modified_str deleted_str summary=""

  if ((${#added[@]} > 0)); then
    added_str="add: ${added[*]}"
    summary+="${added_str}\n"
  fi
  if ((${#modified[@]} > 0)); then
    modified_str="modify: ${modified[*]}"
    summary+="${modified_str}\n"
  fi
  if ((${#deleted[@]} > 0)); then
    deleted_str="delete: ${deleted[*]}"
    summary+="${deleted_str}\n"
  fi

  if [[ -z "$commit_message" ]]; then
    commit_message=$(date +"%Y-%m-%d %H:%M:%S")
  fi

  local full_message="$commit_message"
  [[ -n "$summary" ]] && full_message+="

${summary}"

  # 🖨️ Show summary
  echo
  print -P "%F{cyan}${commit_message}%f"
  ((${#added[@]} > 0)) && print -P "%F{green}add:%f ${added[*]}"
  ((${#modified[@]} > 0)) && print -P "%F{yellow}modify:%f ${modified[*]}"
  ((${#deleted[@]} > 0)) && print -P "%F{red}delete:%f ${deleted[*]}"
  echo

  # ✅ Commit and push
  git commit -m "$full_message"

  local push_output pr_url
  push_output=$(git push --force 2>&1 | tee /dev/tty)

  # 🚀 Auto-open PR if it doesn't already exist
  if echo "$push_output" | grep -q "Create a pull request for"; then
    pr_url=$(echo "$push_output" | grep -Eo 'https://github\.com/[^ ]+')
    if [[ -n "$pr_url" ]]; then
      echo "[gitpush] 🚀 Opening pull request in browser..."
      open "$pr_url"
    fi
  else
    echo "[gitpush] ✅ Push complete. Pull request already exists."
  fi
}
