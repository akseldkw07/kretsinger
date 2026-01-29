# Micromamba hook for bash

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
export MAMBA_EXE="${MAMBA_EXE:-/usr/local/bin/micromamba}"

if [[ -x "$MAMBA_EXE" ]]; then
  __mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
  if [[ $? -eq 0 ]]; then
    eval "$__mamba_setup"
  else
    alias micromamba="$MAMBA_EXE"
  fi
  unset __mamba_setup
fi
