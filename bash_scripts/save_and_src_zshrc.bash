#!/usr/bin/env bash

save_and_src_bashrc() {
  local dest_dir="${KRET}/backup"
  local rc_file="$HOME/.bashrc"          # NOTE: changed from .zshrc
  local backup_file="${dest_dir}/.bashrc"
  local timestamp
  timestamp="$(date +"%Y-%m-%d %H:%M:%S")"
  local original_dir="$PWD"

  mkdir -p "$dest_dir"

  echo "[save_and_src] 🔁 Reloading .bashrc..."
  # shellcheck source=/dev/null
  if ! source "$rc_file"; then
    echo "[save_and_src] ❌ Failed to source .bashrc. Aborting."
    return 1
  fi
  echo "[save_and_src] ✅ .bashrc reloaded."

  if cmp -s "$rc_file" "$backup_file"; then
    echo "[save_and_src] ℹ️ No changes to back up. Exiting."
    return 0
  fi

  cp -f "$rc_file" "$backup_file"

  (
    trap 'cd "$original_dir"' EXIT
    cd "$dest_dir" || exit 1

    echo "[save_and_src] 📋 Git diff for .bashrc:"
    git --no-pager diff --color -- .bashrc || echo "[.bashrc] No visible diff."

    git add .bashrc
    git commit -m "Update .bashrc backup on ${timestamp}" --quiet
    git push --quiet
    echo "[save_and_src] ✅ Backup committed and pushed."
  )
}

# Run on shell exit (bash equivalent of zshexit hook)
trap 'save_and_src_bashrc' EXIT
