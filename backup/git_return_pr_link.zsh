#!/bin/zsh

get_pr_url() {
    local token="$GITHUB_TOKEN" # Set this in your environment
    local repo_url branch owner repo api_url pr_url
    repo_url=$(git config --get remote.origin.url)
    branch=$(git rev-parse --abbrev-ref HEAD)

    # Extract owner and repo from URL (robust for both SSH and HTTPS)
    if [[ "$repo_url" =~ github.com[:/]+([^/]+)/([^/.]+)(\.git)?$ ]]; then
        owner="${match[1]}"
        repo="${match[2]}"
        # Strip .git if present
        repo="${repo%.git}"
    else
        echo "❌ Could not parse GitHub repo URL: $repo_url"
        return 1
    fi

    # Fetch all open PRs and filter locally for the current branch using Python for robust JSON handling
    api_url="https://api.github.com/repos/$owner/$repo/pulls?state=open&per_page=100"
    echo "[DEBUG] API URL: $api_url"
    pr_url=$(
        python3 <<END
import os
import sys
import json
import urllib.request

token = os.environ.get('GITHUB_TOKEN')
if not token:
    sys.exit(1)
req = urllib.request.Request('$api_url', headers={'Authorization': 'token ' + token})
with urllib.request.urlopen(req) as resp:
    data = json.load(resp)
branch = '$branch'
for pr in data:
    if pr.get('head', {}).get('ref') == branch:
        print(pr.get('html_url', ''))
        break
END
    )

    if [[ -n "$pr_url" ]]; then
        echo "$pr_url"
    else
        echo "ℹ️ No open pull request found for branch '$branch'."
    fi
}
