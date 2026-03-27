# ~/.zshrc

# --- History ---
HISTSIZE=1000
SAVEHIST=2000
HISTFILE=~/.zsh_history
setopt histignoredups histignorespace appendhistory

# --- Prompt ---
autoload -Uz promptinit
promptinit
PS1='%n@%m:%~%# '

# --- Key variables ---
export KRET="$HOME/kretsinger"
export PY312_ENV="kret_312"

# --- Homebrew ---
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"

# --- Conda ---
__conda_setup="$('/opt/conda/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        . "/opt/conda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/conda/bin:$PATH"
    fi
fi
unset __conda_setup

# --- Micromamba ---
export MAMBA_EXE="$HOME/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
__mamba_setup="$("$MAMBA_EXE" shell hook --shell zsh --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"
fi
unset __mamba_setup

# --- PATH ---
export PATH="$HOME/.local/bin:$PATH"

# --- Source zsh scripts ---
for script in "$KRET"/zsh_scripts/*.zsh; do
    [ -f "$script" ] && source "$script"
done
