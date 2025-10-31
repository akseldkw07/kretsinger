#!/bin/zsh

# Automate everything about git push, including commit message generation, push, and pull request handling.
# Opens pull request in browser if it doesn't exist, or focuses existing tab if it does.
# Works on systems with `gh` CLI installed, otherwise falls back to manual PR creation.

gitpush() {
  local commit_message="$1"
  git add .

  # Get commit message/description (AI or fallback)
  local summary
  summary=$(_generate_commit_message)

  if [[ -z "$commit_message" ]]; then
    commit_message=$(date +"%Y-%m-%d %H:%M:%S")
  fi

  local full_message="$commit_message"
  [[ -n "$summary" ]] && full_message+="$summary"

  # 🖨️ Show summary
  echo
  print -P "%F{cyan}${commit_message}%f"
  echo "$summary"
  echo

  # Perform commit and push
  _commit_and_push "$full_message"
}

# Perform the git commit and push operations
_commit_and_push() {
  local full_message="$1"

  # ✅ Commit and only push if commit succeeds
  if git commit -m "$full_message"; then
    # Push first and capture output
    local push_output
    push_output=$(git push --force 2>&1)
    if [[ $? -eq 0 ]]; then
      echo "[gitpush] ✅ Push complete."

      # Handle PR creation/opening, passing the push output
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

# Handle pull request creation or opening existing PR
_handle_pull_request() {
  local push_output="$1"

  if command -v gh >/dev/null 2>&1; then
    echo "[gitpush] ✅ GitHub CLI found."
    local current_branch existing_pr pr_url
    current_branch=$(git symbolic-ref --quiet --short HEAD 2>/dev/null)
    existing_pr=$(gh pr list --head "$current_branch" --json number --jq '.[0].number' 2>/dev/null)

    if [[ -n "$existing_pr" ]]; then
      echo "[gitpush] 📋 Pull request already exists (#$existing_pr)."
      pr_url=$(gh pr view "$existing_pr" --json url --jq '.url' 2>/dev/null)
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

# Fallback PR creation using git push output
_fallback_pr_creation() {
  local branch="$1"
  local push_output="$2"
  local pr_url

  echo "[gitpush] 🔍 Checking for PR creation URL..."

  # Check for existing PR URL first
  if echo "$push_output" | grep -q "github.com.*pull"; then
    pr_url=$(echo "$push_output" | grep -Eo 'https://github\.com/[^/]+/[^/]+/pull/[0-9]+')
    if [[ -n "$pr_url" ]]; then
      echo "[gitpush] 🔗 Found existing pull request URL: $pr_url"
      _focus_existing_pr_tab "$pr_url"
      return
    fi
  fi

  # Check for new PR creation URL
    if echo "$push_output" | grep -q "Create a pull request for"; then
    pr_url=$(echo "$push_output" | grep -Eo 'https://github\.com/[^ ]+')
    if [[ -n "$pr_url" ]]; then
      echo "[gitpush] 🚀 Opening pull request creation page in browser..."
      open -a "browser" "$pr_url"
      return
    fi
  fi

  # Try to find existing PR using GitHub API
  echo "[gitpush] 🔍 Checking for existing PR via GitHub API..."
  if pr_url=$(get_pr_url); then
    echo "[gitpush] 🔗 Found existing PR via API: $pr_url"
    _focus_existing_pr_tab "$pr_url"
    return
  fi

  echo "[gitpush] ℹ️  No PR creation URL or existing PR found. You may need to create the PR manually."
}

# Try to focus existing browser tab/window with PR, don't open new tab if not found
_focus_existing_pr_tab() {
  local pr_url="$1"

  if command -v osascript >/dev/null 2>&1; then
    # First check if we can access the browser app at all
    browser_app="Google Chrome"
    if ! osascript -e "tell application \"$browser_app\" to get name" &>/dev/null; then
      echo "[gitpush] ℹ️  ${browser_app} access denied. Please grant permission in System Preferences → Security & Privacy → Privacy → Automation"
      echo "[gitpush] ℹ️  PR URL: $pr_url"
      return
    fi

    local found_tab
    found_tab=$(osascript -e "
      tell application \"$browser_app\"
        try
          if not (exists window 1) then return \"no_windows\"
          set targetURL to \"$pr_url\"
          set baseTargetURL to targetURL
          if baseTargetURL contains \"?\" then
            set baseTargetURL to text 1 thru ((offset of \"?\" in baseTargetURL) - 1) of baseTargetURL
          end if
          set winIdx to 1
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
            set winIdx to winIdx + 1
          end repeat
          return \"not_found\"
        on error errMsg
          return \"error: \" & errMsg
        end try
      end tell
    " 2>/dev/null)

    if [[ "$found_tab" == "found" ]]; then
      echo "[gitpush] ✅ Focused existing ${browser_app} tab/window with PR (fuzzy match)."
    elif [[ "$found_tab" == "no_windows" ]]; then
      echo "[gitpush] ℹ️  ${browser_app} is running but has no windows open."
      echo "[gitpush] ℹ️  PR URL: $pr_url"
    elif [[ "$found_tab" =~ ^error: ]]; then
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

# Generate commit message and description, using aicommits if available, else fallback
_generate_commit_message() {
  local ai_message=""
  if command -v aicommits >/dev/null 2>&1; then
    # Try to get AI-generated commit message (auto-accept feedback)
    ai_message=$(aicommits --yes 2>&1)
    if [[ -n "$ai_message" ]]; then
      if echo "$ai_message" | grep -q 'OpenAI API Error'; then
        echo "[gitpush] AI commit message failed: OpenAI API error. Falling back to handcrafted message." >&2
      else
        echo "[gitpush] Using AI-generated commit message." >&2
        echo "$ai_message"
        return 0
      fi
    else
      echo "[gitpush] AI commit message failed, using fallback." >&2
    fi
  else
    echo "[gitpush] aicommits not found, using fallback." >&2
  fi

  # Fallback: handcrafted logic
  local added=()
  local modified=()
  local deleted=()
  local diff_output
  diff_output=$(git diff --cached --name-status)
  while IFS=$'\t' read -r change_type file; do
    filename="${file##*/}"
    case "$change_type" in
    A) added+=("$filename") ;;
    M) modified+=("$filename") ;;
    D) deleted+=("$filename") ;;
    esac
  done <<<"$diff_output"
  local summary=""
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
  echo "$summary"
}
