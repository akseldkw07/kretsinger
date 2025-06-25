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
  done <<< "$diff_output"

  summary=""
  [[ ${#added[@]} -gt 0 ]] && summary+="add: ${(j: | :)added}
"
  [[ ${#modified[@]} -gt 0 ]] && summary+="modify: ${(j: | :)modified}
"
  [[ ${#deleted[@]} -gt 0 ]] && summary+="delete: ${(j: | :)deleted}
"

  if [[ -z "$commit_message" ]]; then
    commit_message=$(date +"%Y-%m-%d %H:%M:%S")
  fi

  local full_message="${commit_message}"
  [[ -n "$summary" ]] && full_message+="

${summary}"

  echo
  print -P "%F{cyan}${commit_message}%f"
  [[ ${#added[@]} -gt 0 ]] && print -P "%F{green}add:%f ${(j: | :)added}"
  [[ ${#modified[@]} -gt 0 ]] && print -P "%F{yellow}modify:%f ${(j: | :)modified}"
  [[ ${#deleted[@]} -gt 0 ]] && print -P "%F{red}delete:%f ${(j: | :)deleted}"
  echo

  git commit -m "$full_message"
  git push --force
}
