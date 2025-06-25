save_and_src_zshrc() {
    local dest_dir="${KRET}/backup"
    local rc_file="$HOME/.zshrc"
    local backup_file="${dest_dir}/.zshrc"
    local log_file="${dest_dir}/backup_log.txt"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    mkdir -p "$dest_dir"

    # Step 1: Syntax check before continuing
    if ! zsh -n "$rc_file"; then
        echo "[save_and_src_zshrc] ❌ .zshrc has syntax errors. Backup aborted."
        echo "[$timestamp] ❌ .zshrc syntax error detected. No backup made." >>"$log_file"

        # Try to open in default editor or fallback
        if [[ -n "$EDITOR" ]]; then
            echo "[save_and_src_zshrc] Opening .zshrc in \$EDITOR ($EDITOR)..."
            "$EDITOR" "$rc_file"
        else
            echo "[save_and_src_zshrc] No \$EDITOR set. Falling back to 'open'..."
            open "$rc_file"
        fi
        return 1
    fi

    echo "[$timestamp] Manual source triggered." >>"$log_file"
    cp -f "$rc_file" "$backup_file"
    echo "[$timestamp] ✅ .zshrc backed up." >>"$log_file"

    cd "$dest_dir" || return

    if ! git diff --quiet -- .zshrc; then
        echo "[save_and_src_zshrc] 📋 Diff since last backup:"
        git diff --color -- .zshrc

        git add .zshrc
        git commit -m "Backup .zshrc on ${timestamp}" --quiet
        echo "[save_and_src_zshrc] ✅ Changes committed."
    else
        echo "[save_and_src_zshrc] No changes to commit."
    fi

    echo "[save_and_src_zshrc] 🔁 Reloading .zshrc..."
    source "$rc_file"
    echo "[save_and_src_zshrc] ✅ Done."
}
