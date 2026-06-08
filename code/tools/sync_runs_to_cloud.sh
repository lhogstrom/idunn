#!/bin/bash
# Sync runs/ directory to Dropbox
# Usage: ./sync_runs_to_cloud.sh [destination]
#
# Set BALDER_CLOUD_SYNC in your .zshrc:
#   export BALDER_CLOUD_SYNC="$HOME/Dropbox/balder_runs"

set -e

# Use argument, or BALDER_CLOUD_SYNC env var (no fallback default)
DEST="${1:-$BALDER_CLOUD_SYNC}"

if [ -z "$DEST" ]; then
    echo "Error: No destination specified."
    echo ""
    echo "Either:"
    echo "  1. Set BALDER_CLOUD_SYNC in your .zshrc:"
    echo "     export BALDER_CLOUD_SYNC=\"\$HOME/Dropbox/balder_runs\""
    echo ""
    echo "  2. Or pass destination as argument:"
    echo "     ./sync_runs_to_cloud.sh /path/to/destination"
    exit 1
fi

# Find repo root (script is in code/tools/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOURCE="$REPO_ROOT/runs/"

# Validate source exists
if [ ! -d "$SOURCE" ]; then
    echo "Error: Source not found: $SOURCE"
    exit 1
fi

# Create destination if needed
mkdir -p "$DEST"

echo "Syncing runs/ to Dropbox..."
echo "  From: $SOURCE"
echo "  To:   $DEST"
echo ""

# rsync flags:
#   -a: archive mode (preserves everything)
#   -v: verbose
#   -h: human-readable sizes
#   --progress: show transfer progress
#   --delete: mirror source (remove deleted files from dest)
rsync -avh --progress "$SOURCE" "$DEST"

echo ""
echo "Sync complete! $(date)"
