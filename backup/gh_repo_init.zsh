#!/bin/zsh
# Create a GitHub repo from current directory, push, and apply branch protection rulesets.
# Usage: ghinit [-n name] [-d description] [-p] [--no-rules]

ghinit() {
    local repo_name=""
    local description=""
    local visibility="private"
    local branch="main"
    local apply_rules=true

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--name)       repo_name="$2"; shift 2 ;;
            -d|--description) description="$2"; shift 2 ;;
            -p|--public)     visibility="public"; shift ;;
            -b|--branch)     branch="$2"; shift 2 ;;
            --no-rules)      apply_rules=false; shift ;;
            -h|--help)       _ghinit_usage; return 0 ;;
            *)               echo "[ghinit] ❌ Unknown option: $1"; _ghinit_usage; return 1 ;;
        esac
    done

    # Default repo name to current directory name
    if [[ -z "$repo_name" ]]; then
        repo_name="${PWD##*/}"
        echo "[ghinit] ℹ️  Using directory name as repo name: $repo_name"
    fi

    # Validate gh CLI
    if ! command -v gh &>/dev/null; then
        echo "[ghinit] ❌ GitHub CLI (gh) not installed. Get it from https://cli.github.com"
        return 1
    fi

    if ! gh auth status &>/dev/null; then
        echo "[ghinit] ❌ Not authenticated. Run 'gh auth login' first."
        return 1
    fi

    local gh_user
    gh_user=$(gh api user --jq '.login') || {
        echo "[ghinit] ❌ Failed to get GitHub username"
        return 1
    }

    # Initialize git if needed
    _ghinit_setup_git "$branch" || return 1

    # Create initial commit if empty
    _ghinit_initial_commit "$repo_name" "$description" || return 1

    # Create GitHub repo
    echo "[ghinit] 🚀 Creating GitHub repo: $gh_user/$repo_name ($visibility)..."

    local -a gh_args=("$repo_name" "--$visibility" "--source=." "--remote=origin" "--push")
    [[ -n "$description" ]] && gh_args+=("--description=$description")

    if gh repo create "${gh_args[@]}"; then
        echo "[ghinit] ✅ Repository created and pushed"
    else
        echo "[ghinit] ❌ Failed to create repository"
        return 1
    fi

    # Apply branch protection rules
    if [[ "$apply_rules" == true ]]; then
        _ghinit_apply_ruleset "$gh_user" "$repo_name"
    fi

    # Summary
    echo
    print -P "%F{green}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%f"
    print -P "%F{green}Repository setup complete!%f"
    print -P "%F{green}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%f"
    echo
    echo "  URL:        https://github.com/$gh_user/$repo_name"
    echo "  Visibility: $visibility"
    echo "  Branch:     $branch"
    echo "  Rules:      $( [[ "$apply_rules" == true ]] && echo "applied" || echo "skipped" )"
    echo
}

_ghinit_usage() {
    cat <<EOF
Usage: ghinit [-n name] [-d description] [-p] [--no-rules]

Creates a GitHub repository from the current directory.

Options:
  -n, --name <name>         Repository name (default: current directory name)
  -d, --description <desc>  Repository description
  -p, --public              Make repository public (default: private)
  -b, --branch <branch>     Default branch name (default: main)
  --no-rules                Skip applying branch protection rules
  -h, --help                Show this help

Examples:
  ghinit                              # Use directory name, private
  ghinit -n my-project -d "My app"    # Custom name and description
  ghinit -p                           # Public repo
  ghinit --no-rules                   # Skip ruleset (for free GitHub accounts)
EOF
}

_ghinit_setup_git() {
    local branch="$1"

    if [[ ! -d .git ]]; then
        echo "[ghinit] 📁 Initializing git repository..."
        git init -b "$branch"
        echo "[ghinit] ✅ Git initialized"
    else
        echo "[ghinit] ℹ️  Git repository already exists"
    fi
}

_ghinit_initial_commit() {
    local repo_name="$1"
    local description="$2"

    if ! git rev-parse HEAD &>/dev/null; then
        echo "[ghinit] 📝 Creating initial commit..."

        if [[ ! -f README.md ]]; then
            echo "# $repo_name" > README.md
            [[ -n "$description" ]] && echo -e "\n$description" >> README.md
        fi

        git add -A
        git commit -m "Initial commit"
        echo "[ghinit] ✅ Initial commit created"
    fi
}

_ghinit_apply_ruleset() {
    local gh_user="$1"
    local repo_name="$2"

    echo "[ghinit] 🔒 Applying branch protection ruleset..."

    local ruleset_payload
    ruleset_payload=$(cat <<'EOF'
{
  "name": "main",
  "target": "branch",
  "enforcement": "active",
  "conditions": {
    "ref_name": {
      "include": ["~DEFAULT_BRANCH"],
      "exclude": []
    }
  },
  "rules": [
    {"type": "deletion"},
    {"type": "non_fast_forward"},
    {"type": "required_linear_history"},
    {
      "type": "pull_request",
      "parameters": {
        "allowed_merge_methods": ["merge", "squash", "rebase"],
        "dismiss_stale_reviews_on_push": false,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_approving_review_count": 0,
        "required_review_thread_resolution": false
      }
    }
  ]
}
EOF
)

    if gh api "repos/$gh_user/$repo_name/rulesets" --method POST --input - <<< "$ruleset_payload" &>/dev/null; then
        echo "[ghinit] ✅ Branch protection ruleset applied"
    else
        echo "[ghinit] ⚠️  Failed to apply ruleset (requires GitHub Pro/Team plan)"
    fi
}
