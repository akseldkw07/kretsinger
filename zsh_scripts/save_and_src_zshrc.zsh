#!/bin/zsh

# save_and_src_zshrc.zsh - Save and source .zshrc with backup and git commit

save_and_src_zshrc() {
    local dest_dir="${KRET}/backup"
    local rc_file="$HOME/.zshrc"
    local gitconfig_file="$HOME/.gitconfig"
    local backup_basename
    case "$(uname -s)" in
        Darwin) backup_basename=".zshrc" ;;
        Linux)  backup_basename=".zshrc-linux" ;;
        *)      backup_basename=".zshrc-$(uname -s)" ;;
    esac
    local backup_file="${dest_dir}/${backup_basename}"
    local backup_gitconfig_file="${dest_dir}/.gitconfig"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    local original_dir="$PWD"

    mkdir -p "$dest_dir"

    echo "[save_and_src_zshrc] 🔁 Reloading .zshrc..."
    if ! source "$rc_file"; then
        echo "[save_and_src_zshrc] ❌ Failed to source .zshrc. Aborting."
        return 1
    fi
    echo "[save_and_src_zshrc] ✅ .zshrc reloaded."

    if cmp -s "$rc_file" "$backup_file"; then
        echo "[save_and_src_zshrc] ℹ️ No changes to back up. Exiting."
        return 0
    fi

    cp -f "$rc_file" "$backup_file"
    cp -f "$gitconfig_file" "$backup_gitconfig_file"

    # Git actions with guaranteed return to original directory
    (
        trap 'cd "$original_dir"' EXIT
        cd "$dest_dir" || return

        echo "[save_and_src_zshrc] 📋 Git diff for ${backup_basename}:"
        git --no-pager diff --color -- "$backup_basename" || echo "[${backup_basename}] No visible diff."

        git add "$backup_basename"
        git commit -m "Update ${backup_basename} backup on ${timestamp}" --quiet
        git push --quiet
        echo "[save_and_src_zshrc] ✅ Backup committed and pushed."
    )
}
autoload -Uz add-zsh-hook
add-zsh-hook zshexit save_and_src_zshrc
