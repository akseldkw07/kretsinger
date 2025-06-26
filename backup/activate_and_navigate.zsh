#!/bin/zsh

# Usage: source ~/bin/setup_my_env.zsh [target_directory] [env_name]
sod() {
    # --- Default Configuration ---
    DEFAULT_TARGET_DIR="/Users/Akseldkw/coding/kretsinger"

    # --- Parse Optional Arguments ---
    TARGET_DIR="${1:-$DEFAULT_TARGET_DIR}"
    ENV_NAME="${2:-$PY312_ENV}" # PY312_ENV is defined in .zshrc

    # --- Optional customizations ---
    # (e.g., project-specific aliases can go here)

    echo "--- Setting up Environment for '${ENV_NAME}' ---"

    # Check micromamba availability
    if ! command -v micromamba &>/dev/null; then
        echo "Error: 'micromamba' command not found. Please ensure micromamba is installed."
        return 1
    fi

    # Validate micromamba environment
    if ! micromamba env list | grep -q "${ENV_NAME}"; then
        echo "Error: micromamba environment '${ENV_NAME}' not found."
        return 1
    fi

    # Activate the environment
    echo "Activating micromamba environment: ${ENV_NAME}"
    if ! micromamba activate "${ENV_NAME}"; then
        echo "Failed to activate '${ENV_NAME}'."
        return 1
    fi

    # Navigate to the specified directory
    if [ -d "${TARGET_DIR}" ]; then
        echo "Navigating to: ${TARGET_DIR}"
        if ! cd -P "${TARGET_DIR}"; then
            echo "Failed to cd into '${TARGET_DIR}'"
            return 1
        fi
    else
        echo "Error: Target directory '${TARGET_DIR}' does not exist."
        return 1
    fi

    echo "✅ Environment setup complete. Now in '${ENV_NAME}' at '${PWD}'."
}
