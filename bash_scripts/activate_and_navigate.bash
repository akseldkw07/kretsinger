#!/usr/bin/env bash

sod() {
  local DEFAULT_TARGET_DIR="~/kretsinger"

  local TARGET_DIR="${1:-$DEFAULT_TARGET_DIR}"
  local ENV_NAME="${2:-${PY312_ENV}}"

  echo "--- Setting up Environment for '${ENV_NAME}' ---"

  if ! command -v micromamba >/dev/null 2>&1; then
    echo "Error: 'micromamba' command not found. Please ensure micromamba is installed."
    return 1
  fi

  if ! micromamba env list | grep -q "${ENV_NAME}"; then
    echo "Error: micromamba environment '${ENV_NAME}' not found."
    return 1
  fi

  echo "Activating micromamba environment: ${ENV_NAME}"
  if ! micromamba activate "${ENV_NAME}"; then
    echo "Failed to activate '${ENV_NAME}'."
    return 1
  fi

  if [[ -d "${TARGET_DIR}" ]]; then
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
