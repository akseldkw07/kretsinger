# Aliases (bash-friendly)

# If you use lsd, keep this; otherwise change to `ls`
command -v lsd >/dev/null 2>&1 && alias ls='lsd'

alias l='ls -l'
alias lla='ls -la'
alias lt='ls --tree' 2>/dev/null || true

alias ld="ls -ltd -- */"   # directories only
alias mm='micromamba'
alias home='cd "$HOME"'

# Your previous custom aliases (assumes functions exist in sourced scripts)
alias gitrebase='rebase_squash_conflict'
alias gitnuke='rebase_nuclear_feature'

# Small helpers
tailkret() { less +F "$NB_LOGFILE"; }
rgf() { rg --files -g "**/*$@*" 2>/dev/null; }

col() {
  sod "${DESKTOP}/Columbia/"
  ld
}
