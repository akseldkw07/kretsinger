#!/bin/zsh

jnb() {
    # 🧪 1. Ensure Python environment is active
    if [[ "$CONDA_PREFIX" != "$PY312_ENV" && "$CONDA_PREFIX" != "$PY311_ENV" ]]; then
        echo "[jnb] Activating PY312_ENV..."
        micromamba activate "$PY312_ENV" || {
            echo "[jnb] ❌ Failed to activate micromamba env: $PY312_ENV"
            return 1
        }
    fi

    # 🏗️ 2. Parse and override arguments
    local to="html"
    local show_input="--show-input"
    local nb_file=""
    local extra_args=()

    for arg in "$@"; do
        case "$arg" in
        *.ipynb) nb_file="$arg" ;;
        --to=*) to="${arg#--to=}" ;;
        --to)
            shift
            to="$1"
            ;;
        --hide-input) show_input="--hide-input" ;;
        --show-input) show_input="--show-input" ;;
        *) extra_args+=("$arg") ;;
        esac
    done

    if [[ -z "$nb_file" ]]; then
        echo "Usage: jnb notebook.ipynb [--to html|pdf] [--hide-input] [additional args]"
        return 1
    fi

    # 🌀 3. Run nbconvert with resolved arguments
    echo "[jnb] Converting '$nb_file' to $to..."
    jupyter nbconvert --to "$to" "$show_input" "$nb_file" "${extra_args[@]}"

    # 🚀 4. Open in Chrome if output is HTML
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
