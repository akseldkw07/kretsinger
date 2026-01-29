#!/usr/bin/env bash

jnb() {
  if [[ "$CONDA_PREFIX" != "$PY312_ENV" && "$CONDA_PREFIX" != "$PY311_ENV" ]]; then
    echo "[jnb] Activating PY312_ENV..."
    micromamba activate "$PY312_ENV" || {
      echo "[jnb] ❌ Failed to activate micromamba env: $PY312_ENV"
      return 1
    }
  fi

  local to="html"
  local show_input="--show-input"
  local nb_file=""
  local extra_args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      *.ipynb) nb_file="$1"; shift ;;
      --to=*) to="${1#--to=}"; shift ;;
      --to) shift; to="${1:-html}"; shift ;;
      --hide-input) show_input="--hide-input"; shift ;;
      --show-input) show_input="--show-input"; shift ;;
      *) extra_args+=("$1"); shift ;;
    esac
  done

  if [[ -z "$nb_file" ]]; then
    echo "Usage: jnb notebook.ipynb [--to html|pdf] [--hide-input] [additional args]"
    return 1
  fi

  echo "[jnb] Converting '$nb_file' to $to..."
  jupyter nbconvert --to "$to" "$show_input" "$nb_file" "${extra_args[@]}"

  if [[ "$to" == "html" ]]; then
    local output_file="${nb_file%.ipynb}.html"
    if [[ -f "$output_file" ]]; then
      echo "[jnb] Opening $output_file in Chrome..."
      open -a "Google Chrome" "$output_file"
    else
      echo "[jnb] ⚠️ Could not find generated HTML: $output_file"
    fi
  fi
}
