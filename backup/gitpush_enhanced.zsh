#!/bin/zsh

gitpush() {
  local commit_message="$1"
  git add .

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

  local full_message="${commit_message}"
  [[ -n "$summary" ]] && full_message+="

${summary}"

  echo
  print -P "%F{cyan}${commit_message}%f"

  if ((${#added[@]} > 0)); then
    print -P "%F{green}add:%f ${added[*]}"
  fi
  if ((${#modified[@]} > 0)); then
    print -P "%F{yellow}modify:%f ${modified[*]}"
  fi
  if ((${#deleted[@]} > 0)); then
    print -P "%F{red}delete:%f ${deleted[*]}"
  fi
  echo

  git commit -m "$full_message"
  git push --force
}
