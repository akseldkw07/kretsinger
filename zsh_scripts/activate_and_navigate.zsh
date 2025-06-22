#!/bin/zsh

# This script activates a specific micromamba environment and navigates to a project directory.
# It should be SOURCED, not executed directly.
# Example: source ~/bin/setup_my_env.zsh

# --- User Configuration ---
# 1. Name of your micromamba environment
ENV_NAME="kret_312"

# 2. Path to your project directory
TARGET_DIR='/Users/Akseldkw/coding/kretsinger'

# 3. (Optional) Define any specific shell commands, aliases, or functions
#    that you want to be available *only* when this environment is active
#    and you are in this directory.

# Example:
# ALIAS_NAME="run_dev_server"
# ALIAS_COMMAND="python app.py --env dev"
# alias ${ALIAS_NAME}="${ALIAS_COMMAND}"
# echo "Alias '${ALIAS_NAME}' created to run: ${ALIAS_COMMAND}"
# -------------------------

echo "--- Setting up Environment for ${ENV_NAME} ---"

# Check if micromamba command is available
if ! command -v micromamba &> /dev/null; then
    echo "Error: 'micromamba' command not found. Please ensure micromamba is installed and initialized for your shell."
    return 1 # 'return' is used in sourced scripts instead of 'exit'
fi

# Check if the micromamba environment exists
if ! micromamba env list | grep -q "${ENV_NAME}"; then
    echo "Error: micromamba environment '${ENV_NAME}' not found. Please create it first."
    return 1
fi

# Activate the micromamba environment
# Using 'micromamba activate' directly modifies the current shell's environment.
echo "Activating micromamba environment: ${ENV_NAME}"
if ! micromamba activate "${ENV_NAME}"; then
    echo "Failed to activate '${ENV_NAME}'. Check micromamba installation and environment name."
    return 1
fi

# Navigate to the specified location
if [ -d "${TARGET_DIR}" ]; then
    echo "Navigating to: ${TARGET_DIR}"
    # Use 'cd -P' to resolve symlinks and ensure canonical path
    if ! cd -P "${TARGET_DIR}"; then
        echo "Failed to change directory to '${TARGET_DIR}'. Check path permissions."
        return 1
    fi
else
    echo "Error: Target directory '${TARGET_DIR}' does not exist. Please create it or update the script."
    return 1
fi

echo "Environment setup complete! You are now in the '${ENV_NAME}' environment at '${PWD}'."
