# Google Cloud SDK (bash completion + path)
GCLOUD_DIR="$HOME/google-cloud-sdk"

if [[ -f "$GCLOUD_DIR/path.bash.inc" ]]; then
  # shellcheck source=/dev/null
  . "$GCLOUD_DIR/path.bash.inc"
fi

if [[ -f "$GCLOUD_DIR/completion.bash.inc" ]]; then
  # shellcheck source=/dev/null
  . "$GCLOUD_DIR/completion.bash.inc"
fi
