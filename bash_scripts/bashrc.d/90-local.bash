# Machine-local overrides (NOT tracked)
# Create ~/.bashrc.local to override per-machine stuff:
# - KRET path
# - extra PATH stuff
# - secrets sourcing
# - host-specific aliases

if [[ -f "$HOME/.bashrc.local" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/.bashrc.local"
fi
