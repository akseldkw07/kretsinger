#!/bin/zsh

kretsave() {
    local src_dir="${KRET}/zsh_scripts"
    local dest_dir="${KRET}/backup"
    local rc_file="$HOME/.zshrc"
    local log_file="${dest_dir}/backup_log.txt"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    mkdir -p "$dest_dir"

    echo "[$timestamp] Starting backup..." >>"$log_file"

    rsync -av --delete "$src_dir"/ "$dest_dir"/ >>"$log_file" 2>&1
    rsync -av --update "$rc_file" "$dest_dir"/ >>"$log_file" 2>&1

    echo "[$timestamp] ✅ Backup complete." >>"$log_file"
    # Quick preview at shell exit
    echo "[kretsave] Recent activity:"
    tail -n 3 "$log_file"
}
autoload -Uz add-zsh-hook
add-zsh-hook zshexit backup
