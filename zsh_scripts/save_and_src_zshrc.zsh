save_and_src_zshrc() {
    local dest_dir="${KRET}/backup"
    local rc_file="$HOME/.zshrc"
    local backup_file="${dest_dir}/.zshrc"
    local log_file="${dest_dir}/backup_log.txt"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    mkdir -p "$dest_dir"

    echo "[save_and_src_zshrc] 🔁 Reloading .zshrc..."
    if ! source "$rc_file"; then
        echo "[save_and_src_zshrc] ❌ Failed to source .zshrc. Aborting."
        echo "[$timestamp] ❌ Error sourcing .zshrc. Backup aborted." >>"$log_file"
        return 1
    fi
    echo "[save_and_src_zshrc] ✅ .zshrc reloaded."

    # If no changes, just log and exit
    if cmp -s "$rc_file" "$backup_file"; then
        echo "[$timestamp] ℹ️ No changes detected in .zshrc. Early exit." >>"$log_file"
        echo "[save_and_src_zshrc] ℹ️ No changes to back up. Exiting."
        return 0
    fi

    # Back up the updated .zshrc
    cp -f "$rc_file" "$backup_file"
    echo "[$timestamp] ✅ .zshrc updated and backed up." >>"$log_file"

    # Only do Git actions if current directory is within $KRET
    if [[ "$PWD" == "$KRET" || "$PWD" == "$KRET"/* ]]; then
        echo "[save_and_src_zshrc] 📋 Git diff for .zshrc:"
        git -C "$dest_dir" diff --color -- .zshrc || echo "[.zshrc] No visible diff."

        git -C "$dest_dir" add .zshrc
        git -C "$dest_dir" commit -m "Update .zshrc backup on ${timestamp}" --quiet
        git -C "$dest_dir" push --quiet
        echo "[save_and_src_zshrc] ✅ Backup committed and pushed."
    else
        echo "[save_and_src_zshrc] ℹ️ Git actions skipped (outside KRET directory tree)."
    fi
}
