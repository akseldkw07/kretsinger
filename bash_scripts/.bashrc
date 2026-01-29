# ~/.bashrc (symlinked here)

# If not interactive, don't do interactive things
case $- in
  *i*) ;;
  *) return ;;
esac

# Helpful bash behavior
shopt -s histappend
shopt -s checkwinsize

# Source modular config files (sorted)
DOTFILES_BASH_DIR="${DOTFILES_BASH_DIR:-$HOME/.dotfiles/bash}"
BASHRC_D="$DOTFILES_BASH_DIR/bashrc.d"

if [[ -d "$BASHRC_D" ]]; then
  while IFS= read -r -d '' f; do
    # shellcheck source=/dev/null
    source "$f"
  done < <(find "$BASHRC_D" -type f -name '*.bash' -print0 | sort -z)
fi
