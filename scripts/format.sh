#!/usr/bin/env bash
set -euo pipefail

# Format all C/C++ sources in the repo, respecting .gitignore and skipping thirdparty/
# Usage:
#   ./format.sh          # format in place
#   ./format.sh --check  # only check, fail on diffs

mode="fix"
if [[ "${1:-}" == "--check" || "${1:-}" == "-n" ]]; then
  mode="check"
fi

if ! command -v clang-format >/dev/null 2>&1; then
  os_name=$(uname -s)
  if [[ "$os_name" == "Darwin" ]]; then
    echo "error: clang-format not found. Install it with: brew install clang-format" >&2
  else
    echo "error: clang-format not found. Install it with your package manager, e.g.: sudo apt-get install clang-format" >&2
  fi
  exit 1
fi

# Collect tracked and untracked (but not ignored) files with C/C++ extensions
# Respect .gitignore via --exclude-standard; filter out thirdparty/
files=$(git ls-files --cached --others --exclude-standard -- \
  '*.h' '*.hh' '*.hpp' '*.hxx' \
  '*.c' '*.cc' '*.cpp' '*.cxx' | grep -v '^thirdparty/')

if [[ -z "$files" ]]; then
  echo "No source files to format."
  exit 0
fi

if [[ "$mode" == "check" ]]; then
  # Newer clang-format supports --dry-run -Werror; fallback to diff otherwise
  if clang-format --version | grep -Eq 'version (1[1-9]|[2-9][0-9])'; then
    echo "$files" | xargs clang-format --dry-run -Werror
    exit $?
  else
    rc=0
    tmpdir=$(mktemp -d)
    trap 'rm -rf "$tmpdir"' EXIT
    OLDIFS=$IFS; IFS=$'\n'
    for f in $files; do
      clang-format "$f" >"$tmpdir/out" || true
      if ! diff -q "$f" "$tmpdir/out" >/dev/null; then
        echo "Would reformat: $f"
        rc=1
      fi
    done
    IFS=$OLDIFS
    exit $rc
  fi
else
  echo "$files" | xargs clang-format -i
  echo "Formatted $(echo "$files" | wc -l | tr -d ' ') files."
fi
