#!/usr/bin/env bash
# Refresh training/launchers_mirror/ from the internvl-u submodule (read-only snapshot for GitHub browsing).
set -euo pipefail
BP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$BP/training/models/internvl-u/internvl_chat/shell/internvlu"
DST="$BP/training/launchers_mirror"
rm -rf "$DST"; mkdir -p "$DST"
for d in sft orchestration engine; do cp -r "$SRC/$d" "$DST/$d"; done
cp "$SRC/README.md" "$DST/SUBMODULE_README.md"
SHA="$(git -C "$BP/training/models/internvl-u" rev-parse HEAD)"
cat > "$DST/README.md" <<EOT
# launchers_mirror (read-only snapshot)

Copy of \`training/models/internvl-u/internvl_chat/shell/internvlu/{sft,orchestration,engine}/\`
at submodule commit \`$SHA\`, so the training launchers are browsable on GitHub (the submodule
URL is a local path that GitHub cannot resolve).

**Do not edit or run these copies.** The scripts resolve paths relative to their location inside the
submodule (\`REPO_ROOT\`, \`BP_ROOT\`), so they only work there. Edit the submodule, then run
\`training/sync_launchers_mirror.sh\` to refresh this mirror.
EOT
echo "mirrored @ $SHA"
