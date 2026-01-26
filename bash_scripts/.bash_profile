# ~/.bash_profile (symlinked here)
# macOS Terminal/iTerm often launches login shells.
# Keep it simple: always source ~/.bashrc

if [[ -f "$HOME/.bashrc" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/.bashrc"
fi
