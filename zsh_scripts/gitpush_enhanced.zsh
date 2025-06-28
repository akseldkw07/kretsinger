#!/bin/zsh

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
  echo "_handle_pull_request!!!!!!!!!!!"
  echo "$push_output"

  if command -v gbw >/dev/null 2>&1; then
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
      echo "[gitpush] � Found existing pull request URL: $pr_url"
      echo "[gitpush] 🔗 PR URL: $pr_url"
      _focus_existing_pr_tab "$pr_url"
      return
    fi
  fi

  # Check for new PR creation URL
  if echo "$push_output" | grep -q "Create a pull request for"; then
    pr_url=$(echo "$push_output" | grep -Eo 'https://github\.com/[^ ]+')
    if [[ -n "$pr_url" ]]; then
      echo "[gitpush] 🚀 Opening pull request creation page in browser..."
      open "$pr_url"
      return
    fi
  fi

  # If no PR link found, try to find an existing PR for this branch using GitHub web UI
  if [[ -n "$branch" ]]; then
    # Get repo info from git remote
    local remote_url repo_path pr_list_url
    remote_url=$(git config --get remote.origin.url)
    # Handle both git@github.com: and https://github.com/ URLs
    if [[ "$remote_url" =~ github.com[:/](.+)\.git ]]; then
      repo_path=${BASH_REMATCH[1]}
      pr_list_url="https://github.com/$repo_path/pulls?q=is%3Apr+is%3Aopen+head%3A$branch"
      echo "[gitpush] 🔍 Opening PR list for branch in browser: $pr_list_url"
      open "$pr_list_url"
      return
    fi
  fi

  echo "[gitpush] ℹ️  No PR creation URL or existing PR found. You may need to create the PR manually."
}

# Try to focus existing Chrome tab with PR, fallback to not opening if fails
_focus_existing_pr_tab() {
  local pr_url="$1"

  if command -v osascript >/dev/null 2>&1; then
    osascript -e "
      tell application \"Google Chrome\"
        set theURL to \"$pr_url\"
        repeat with theWindow in windows
          repeat with theTab in tabs of theWindow
            if URL of theTab contains \"github.com\" and URL of theTab contains \"pull\" then
              set active tab index of theWindow to index of theTab
              set index of theWindow to 1
              activate
              return
            end if
          end repeat
        end repeat
      end tell
    " 2>/dev/null || echo "[gitpush] ℹ️  PR exists but couldn't focus existing tab."
  else
    echo "[gitpush] ℹ️  PR exists. URL logged above."
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
