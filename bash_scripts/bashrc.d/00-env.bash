# Core env vars (portable defaults)

export KRET="${KRET:-$HOME/coding/kretsinger}"
export PY312_ENV="${PY312_ENV:-kret_312}"
export PY311_ENV="${PY311_ENV:-kret_311}"

export MM_PATH="${MM_PATH:-$HOME/micromamba/envs}"
export PY312_PATH="${PY312_PATH:-${MM_PATH}/${PY312_ENV}/bin/python}"
export PY311_PATH="${PY311_PATH:-${MM_PATH}/${PY311_ENV}/bin/python}"

export NB_LOGFILE="${NB_LOGFILE:-${KRET}/data/nb_log.log}"
export DESKTOP="${DESKTOP:-$HOME/Desktop}"

# Convenience PATH additions (safe to repeat)
export PATH="$HOME/.local/bin:$PATH"
