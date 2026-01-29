#!/usr/bin/env bash

gitpush() {
  local commit_message="$1"
  git add .

  local summary
  summary="$(_generate_commit_message)"

  if [[ -z "$commit_message" ]]; then
    commit_message="$(date +"%Y-%m-%d %H:%M:%S")"
  fi

  local full_message="$commit_message"
  [[ -n "$summary" ]] && full_message+="$summary"

  echo
  printf "%s\n" "$commit_message"
  printf "%s\n" "$summary"
  echo

  _commit_and_push "$full_message"
}

_commit_and_push() {
  local full_message="$1"

  if git commit -m "$full_message"; then
    local push_output
    push_output="$(git push --force 2>&1)"
    if [[ $? -eq 0 ]]; then
      echo "[gitpush] ✅ Push complete."
      _handle_pull_request "$push_output"
    else
      echo "[gitpush] ❌ Push failed."
      return 1
    fi
  else
    echo "[gitpush] ❌ Commit failed. Push aborted."
    return 1
  fi
}

_handle_pull_request() {
  local push_output="$1"

  if command -v gh >/dev/null 2>&1; then
    echo "[gitpush] ✅ GitHub CLI found."
    local current_branch existing_pr pr_url
    current_branch="$(git symbolic-ref --quiet --short HEAD 2>/dev/null)"
    existing_pr="$(gh pr list --head "$current_branch" --json number --jq '.[0].number' 2>/dev/null)"

    if [[ -n "$existing_pr" ]]; then
      echo "[gitpush] 📋 Pull request already exists (#$existing_pr)."
      pr_url="$(gh pr view "$existing_pr" --json url --jq '.url' 2>/dev/null)"
      if [[ -n "$pr_url" ]]; then
        echo "[gitpush] 🔗 PR URL: $pr_url"
        _focus_existing_pr_tab "$pr_url"
      fi
    else
      echo "[gitpush] 🆕 Creating new pull request..."
      if gh pr create --fill --web >/dev/null 2>&1; then
        echo "[gitpush] 🚀 Pull request created and opened in browser."
      else
        echo "[gitpush] ❌ Failed to create pull request via gh CLI."
        _fallback_pr_creation "$current_branch" "$push_output"
      fi
    fi
  else
    echo "[gitpush] ❌ GitHub CLI not found."
    _fallback_pr_creation "" "$push_output"
  fi
}

_fallback_pr_creation() {
  local branch="$1"
  local push_output="$2"
  local pr_url

  echo "[gitpush] 🔍 Checking for PR creation URL..."

  if echo "$push_output" | grep -q "github.com.*pull"; then
    pr_url="$(echo "$push_output" | grep -Eo 'https://github\.com/[^/]+/[^/]+/pull/[0-9]+')"
    if [[ -n "$pr_url" ]]; then
      echo "[gitpush] 🔗 Found existing pull request URL: $pr_url"
      _focus_existing_pr_tab "$pr_url"
      return
    fi
  fi

  if echo "$push_output" | grep -q "Create a pull request for"; then
    pr_url="$(echo "$push_output" | grep -Eo 'https://github\.com/[^ ]+')"
    if [[ -n "$pr_url" ]]; then
      echo "[gitpush] 🚀 Opening pull request creation page in browser..."
      open -a "browser" "$pr_url"
      return
    fi
  fi

  echo "[gitpush] 🔍 Checking for existing PR via GitHub API..."
  if pr_url="$(get_pr_url)"; then
    echo "[gitpush] 🔗 Found existing PR via API: $pr_url"
    _focus_existing_pr_tab "$pr_url"
    return
  fi

  echo "[gitpush] ℹ️  No PR creation URL or existing PR found. You may need to create the PR manually."
}

_focus_existing_pr_tab() {
  local pr_url="$1"

  if command -v osascript >/dev/null 2>&1; then
    local browser_app="Google Chrome"
    if ! osascript -e "tell application \"$browser_app\" to get name" >/dev/null 2>&1; then
      echo "[gitpush] ℹ️  ${browser_app} access denied. Grant permission in System Settings → Privacy & Security → Automation"
      echo "[gitpush] ℹ️  PR URL: $pr_url"
      return
    fi

    local found_tab
    found_tab="$(osascript -e "
      tell application \"$browser_app\"
        try
          if not (exists window 1) then return \"no_windows\"
          set targetURL to \"$pr_url\"
          set baseTargetURL to targetURL
          if baseTargetURL contains \"?\" then
            set baseTargetURL to text 1 thru ((offset of \"?\" in baseTargetURL) - 1) of baseTargetURL
          end if
          repeat with theWindow in windows
            set tabIdx to 1
            repeat with theTab in tabs of theWindow
              set tabURL to URL of theTab
              set baseTabURL to tabURL
              if baseTabURL contains \"?\" then
                set baseTabURL to text 1 thru ((offset of \"?\" in baseTabURL) - 1) of baseTabURL
              end if
              if tabURL is equal to targetURL or tabURL starts with targetURL or baseTabURL is equal to baseTargetURL then
                set active tab index of theWindow to tabIdx
                set index of theWindow to 1
                activate
                return \"found\"
              end if
              set tabIdx to tabIdx + 1
            end repeat
          end repeat
          return \"not_found\"
        on error errMsg
          return \"error: \" & errMsg
        end try
      end tell
    " 2>/dev/null)"

    if [[ "$found_tab" == "found" ]]; then
      echo "[gitpush] ✅ Focused existing ${browser_app} tab/window with PR."
    elif [[ "$found_tab" == "no_windows" ]]; then
      echo "[gitpush] ℹ️  ${browser_app} is running but has no windows open."
      echo "[gitpush] ℹ️  PR URL: $pr_url"
    elif [[ "$found_tab" == error:* ]]; then
      echo "[gitpush] ℹ️  ${browser_app} access error: ${found_tab#error: }"
      echo "[gitpush] ℹ️  PR URL: $pr_url"
    else
      echo "[gitpush] ℹ️  No existing ${browser_app} tab/window found for this PR."
      echo "[gitpush] ℹ️  PR URL: $pr_url"
    fi
  else
    echo "[gitpush] ℹ️  AppleScript not available. PR URL: $pr_url"
  fi
}

_generate_commit_message() {
  local added=()
  local modified=()
  local deleted=()
  local diff_output
  diff_output="$(git diff --cached --name-status)"

  while IFS=$'\t' read -r change_type file; do
    [[ -z "$change_type" ]] && continue
    local filename="${file##*/}"
    case "$change_type" in
      A) added+=("$filename") ;;
      M) modified+=("$filename") ;;
      D) deleted+=("$filename") ;;
    esac
  done <<<"$diff_output"

  local summary=""
  if ((${#added[@]} > 0)); then
    summary+=$'\nAdded:'
    for file in "${added[@]}"; do summary+=$'\n- '"$file"; done
  fi
  if ((${#modified[@]} > 0)); then
    summary+=$'\nModified:'
    for file in "${modified[@]}"; do summary+=$'\n- '"$file"; done
  fi
  if ((${#deleted[@]} > 0)); then
    summary+=$'\nDeleted:'
    for file in "${deleted[@]}"; do summary+=$'\n- '"$file"; done
  fi

  echo "$summary"
}
