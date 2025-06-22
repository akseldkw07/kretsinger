#!/bin/bash

# This script automates Git operations:
# 1. Dynamically determines the primary branch name (e.g., 'main' or 'master').
# 2. Checks out the primary branch.
# 3. Pulls the latest changes from the primary branch.
# 4. Identifies and offers to delete all other local branches.

# --- Script Logic ---

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Git Branch Cleanup Script ---"

# Check if the current directory is a Git repository
if ! git rev-parse --is-inside-work-tree &> /dev/null; then
    echo "Error: Not inside a Git repository. Please navigate to your repository."
    exit 1
fi

# Dynamically determine the primary branch name
# First, try to get the default branch from the 'origin' remote's HEAD.
# This is usually the branch that new clones default to.
PRIMARY_BRANCH=$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null)

if [ -z "$PRIMARY_BRANCH" ]; then
    # If origin/HEAD is not set or remote 'origin' doesn't exist,
    # fallback to checking for common primary branch names locally.
    if git rev-parse --verify main &>/dev/null; then
        PRIMARY_BRANCH="main"
    elif git rev-parse --verify master &>/dev/null; then
        PRIMARY_BRANCH="master"
    else
        echo "Error: Could not determine primary branch (tried 'main', 'master', and remote 'origin/HEAD')."
        echo "Please ensure you have a 'main' or 'master' branch, or set the PRIMARY_BRANCH variable manually in the script."
        exit 1
    fi
else
    # Extract the simple branch name (e.g., 'main' from 'origin/main')
    PRIMARY_BRANCH="${PRIMARY_BRANCH##*/}"
fi

echo "Determined primary branch: '${PRIMARY_BRANCH}'."

# 1. Checkout the primary branch
echo "Checking out the '${PRIMARY_BRANCH}' branch..."
if ! git checkout "${PRIMARY_BRANCH}"; then
    echo "Error: Failed to checkout '${PRIMARY_BRANCH}'. Please ensure it exists and is accessible."
    exit 1
fi
echo "Successfully switched to '${PRIMARY_BRANCH}'."

# 2. Pull latest changes from the primary branch
echo "Pulling latest changes from '${PRIMARY_BRANCH}'..."
# Ensure 'origin' remote exists before attempting to pull from it
if git remote -v | grep -q '^origin'; then
    if ! git pull origin "${PRIMARY_BRANCH}"; then
        echo "Warning: Failed to pull latest from 'origin/${PRIMARY_BRANCH}'. You might have local changes, network issues, or a different upstream."
        # We won't exit here, as deleting other branches might still be desired.
    fi
else
    echo "Warning: No 'origin' remote found. Skipping pull from remote."
fi
echo "Pulled latest from '${PRIMARY_BRANCH}' (if remote was available)."


# 3. Identify and delete other local branches
echo ""
echo "Identifying non-${PRIMARY_BRANCH} local branches for deletion..."

# Get a list of all local branches, excluding the primary branch
# 'git branch' lists local branches.
# 'grep -v "^*"' excludes the current branch (just in case, though we're on primary).
# 'grep -v "${PRIMARY_BRANCH}"' excludes the primary branch.
# 'sed "s/^[[:space:]]*//"' removes leading spaces.
BRANCHES_TO_DELETE=$(git branch | grep -v "^*" | grep -v "${PRIMARY_BRANCH}" | sed "s/^[[:space:]]*//")

if [ -z "$BRANCHES_TO_DELETE" ]; then
    echo "No other local branches found to delete."
else
    echo "The following branches will be considered for deletion:"
    echo "${BRANCHES_TO_DELETE}"
    echo ""

    # Loop through each branch and ask for confirmation before deleting
    for branch in ${BRANCHES_TO_DELETE}; do
        read -r -p "Delete local branch '${branch}'? (y/N): " CONFIRMATION
        if [[ "$CONFIRMATION" =~ ^[Yy]$ ]]; then
            echo "Deleting branch '${branch}'..."
            # -d: delete the branch if it has been fully merged upstream.
            # -D: force delete the branch, even if it hasn't been merged.
            # We'll use -d for safety, you can change to -D if you understand the risks.
            if git branch -D "${branch}"; then
                echo "Successfully deleted '${branch}'."
            else
                echo "Warning: Could not delete '${branch}'. It might contain unmerged changes. Use 'git branch -D ${branch}' to force delete."
            fi
        else
            echo "Skipping branch '${branch}'."
        fi
    done
fi

echo ""
echo "--- Script Finished ---"
echo "You are currently on the '${PRIMARY_BRANCH}' branch."
echo "Use 'git branch' to see your remaining local branches."
