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
      # Before creating a fresh PR, try to focus an already-open PR/compare tab in the default browser
      if _focus_existing_repo_pr_or_compare_tab; then
        echo "[gitpush] ℹ️  Found existing browser tab for this repo/branch; not creating a new PR tab."
        return
      fi
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
      open "$pr_url"
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

_focus_existing_repo_pr_or_compare_tab() {
  # Derive owner/repo and branch
  local remote url owner repo branch baseRepo branchComparePath
  remote=$(git remote get-url --push origin 2>/dev/null)
  branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
  if [[ -z "$remote" || -z "$branch" ]]; then
    return 1
  fi
  # Normalize GitHub URL
  # Supports: git@github.com:owner/repo.git or https://github.com/owner/repo(.git)
  if [[ "$remote" == git@github.com:* ]]; then
    url="https://github.com/${remote#git@github.com:}"
  else
    url="$remote"
  fi
  url=${url%.git}
  # Extract owner and repo
  owner=$(echo "$url" | sed -n 's#https://github\.com/\([^/]*\)/\([^/]*\).*#\1#p')
  repo=$(echo "$url" | sed -n 's#https://github\.com/\([^/]*\)/\([^/]*\).*#\2#p')
  if [[ -z "$owner" || -z "$repo" ]]; then
    return 1
  fi
  baseRepo="https://github.com/${owner}/${repo}/"
  branchComparePath="compare/${branch}"

  # If AppleScript isn't available, give up
  if ! command -v osascript >/dev/null 2>&1; then
    return 1
  fi

  # Try known browsers regardless of default (Safari + Chromium family)
  local -a _browsers=(
    com.apple.Safari
    company.thebrowser.Browser
    com.google.Chrome
    com.google.Chrome.canary
    org.chromium.Chromium
    com.brave.Browser
    com.microsoft.edgemac
  )

  local bid result appname
  for bid in "${_browsers[@]}"; do
    case "$bid" in
      com.apple.Safari)
        result=$(_run_applescript_repo "$bid" "$safari_repo_as") ;;
      *)
        result=$(_run_applescript_repo "$bid" "$chromium_repo_as") ;;
    esac
    if [[ "$result" == "found" ]]; then
      appname=$(osascript -e 'name of application id "'"$bid"'"' 2>/dev/null)
      [[ -z "$appname" ]] && appname="$bid"
      echo "[gitpush] 🔎 Focused existing GitHub tab for this repo/branch in $appname."
      return 0
    fi
  done

  return 1
}


_focus_existing_pr_tab() {
  local pr_url="$1"

  # If AppleScript isn't available, just open in default browser
  if ! command -v osascript >/dev/null 2>&1; then
    echo "[gitpush] ℹ️  AppleScript not available. PR URL: $pr_url"
    open "$pr_url"
    return
  fi

  # Try known browsers regardless of default (Safari + Chromium family)
  local -a _browsers=(
    com.apple.Safari
    company.thebrowser.Browser
    com.google.Chrome
    com.google.Chrome.canary
    org.chromium.Chromium
    com.brave.Browser
    com.microsoft.edgemac
  )

  local bid result appname
  for bid in "${_browsers[@]}"; do
    case "$bid" in
      com.apple.Safari)
        result=$(_run_applescript "$bid" "$safari_as") ;;
      *)
        result=$(_run_applescript "$bid" "$chromium_as") ;;
    esac
    if [[ "$result" == "found" ]]; then
      appname=$(osascript -e 'name of application id "'"$bid"'"' 2>/dev/null)
      [[ -z "$appname" ]] && appname="$bid"
      echo "[gitpush] ✅ Focused existing tab in $appname."
      return
    fi
  done

  echo "[gitpush] ℹ️  No existing tab found in known browsers."
  echo "[gitpush] 🔗 Opening PR URL in default browser..."
  open "$pr_url"
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
