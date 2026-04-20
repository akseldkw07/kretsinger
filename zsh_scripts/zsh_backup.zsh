#!/bin/zsh

kretsave() {
    local src_dir="${KRET}/zsh_scripts"
    local dest_dir="${KRET}/backup"
    local rc_file="$HOME/.zshrc"

    mkdir -p "$dest_dir"

    rsync -av --delete "$src_dir"/ "$dest_dir"/
    rsync -av --update "$rc_file" "$dest_dir"/

    echo "[kretsave] ✅ Backup complete."
}
autoload -Uz add-zsh-hook
add-zsh-hook zshexit backup
