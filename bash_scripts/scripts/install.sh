#!/usr/bin/env bash
set -euo pipefail

# Where the repo lives (assumes you run install.sh from inside the repo)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOTFILES_HOME="${DOTFILES_HOME:-$HOME/.dotfiles}"

mkdir -p "$DOTFILES_HOME"

# If you cloned elsewhere, we still want a stable path:
# copy/symlink repo into ~/.dotfiles if you want.
# Minimal approach: just remember where bash lives.
BASH_DIR="$REPO_ROOT/bash"

# Export for bashrc to find modules
# We store it in a small env file sourced by .bashrc.local optionally,
# but easiest: write to ~/.bashrc.local if it doesn't exist.
ensure_local() {
  if [[ ! -f "$HOME/.bashrc.local" ]]; then
    cat >"$HOME/.bashrc.local" <<'EOF'
# Local overrides for this machine (not in git)
# Example:
# export KRET="$HOME/coding/kretsinger"
EOF
  fi

  if ! grep -q 'DOTFILES_BASH_DIR=' "$HOME/.bashrc.local"; then
    printf '\nexport DOTFILES_BASH_DIR="%s"\n' "$BASH_DIR" >>"$HOME/.bashrc.local"
  fi
}

backup_if_needed() {
  local target="$1"
  if [[ -e "$target" && ! -L "$target" ]]; then
    local ts
    ts="$(date +"%Y%m%d-%H%M%S")"
    mv "$target" "${target}.bak-${ts}"
    echo "[install] Backed up $target -> ${target}.bak-${ts}"
  fi
}

link_file() {
  local src="$1"
  local dest="$2"

  backup_if_needed "$dest"
  ln -sfn "$src" "$dest"
  echo "[install] Linked $dest -> $src"
}

main() {
  ensure_local

  link_file "$BASH_DIR/bashrc" "$HOME/.bashrc"
  link_file "$BASH_DIR/bash_profile" "$HOME/.bash_profile"

  echo
  echo "[install] Done."
  echo "[install] Restart your terminal or run: source ~/.bashrc"
}

main "$@"
