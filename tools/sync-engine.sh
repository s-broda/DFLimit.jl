#!/bin/sh
# tools/sync-engine.sh: copy the Julia engine from a DF-Limit checkout into src/ of this package.
#
#   sh tools/sync-engine.sh /path/to/DF-Limit
#
# The script only reads the DF-Limit checkout.  It
#   1. copies the engine, julia/DFExact.jl, and the file it includes, julia/df_series.jl, into src/ (the engine reads
#      no other file at load or run time; the script stops if a copied file includes one that is not copied),
#   2. converts them to LF line endings (a Windows checkout with core.autocrlf=true has CRLF),
#   3. applies the patches in tools/patches/*.patch, if there are any, in name order (a patch is a change that the
#      package requires and that should go upstream into DF-Limit's julia/; once it is upstream, delete the patch
#      file; there are none at present),
#   4. records the upstream commit in UPSTREAM.
# Left out: the scripts and checks of julia/ (quickcheck.jl, test_dfexact.jl, certify_dfexact.jl, ...), its
# Project.toml and Manifest.toml, and julia/replication/.
set -e

if [ $# -ne 1 ]; then echo "usage: sh tools/sync-engine.sh /path/to/DF-Limit" >&2; exit 2; fi
UP=$(cd "$1" && pwd)
PKG=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$UP/julia/DFExact.jl" ] || { echo "$UP does not look like a DF-Limit checkout (no julia/DFExact.jl)" >&2; exit 1; }
grep -q '^name = "DFLimit"' "$PKG/Project.toml" 2>/dev/null || { echo "$PKG is not the package root" >&2; exit 1; }

ENGINE="DFExact.jl df_series.jl"

lf() { tr -d '\r' < "$1" > "$2"; }
for f in $ENGINE; do lf "$UP/julia/$f" "$PKG/src/$f"; done

# every include of the copied files must be one of the copied files
for f in $ENGINE; do
    for inc in $(sed -n 's/^[[:space:]]*include("\([^"]*\)").*/\1/p' "$PKG/src/$f"); do
        case " $ENGINE " in
            *" $inc "*) ;;
            *) echo "src/$f includes $inc, which the script does not copy: add it to ENGINE" >&2; exit 1 ;;
        esac
    done
done

# the patches (unified diffs relative to the package root, paths src/...)
PATCHES=""
for p in "$PKG"/tools/patches/*.patch; do
    [ -f "$p" ] || continue
    patch -p1 -s --forward --no-backup-if-mismatch -d "$PKG" < "$p"
    PATCHES="$PATCHES $(basename "$p")"
done

# the upstream commit (read-only git commands: --no-optional-locks keeps git from refreshing DF-Limit's index)
G="git --no-optional-locks -C $UP"
COMMIT=$($G rev-parse HEAD 2>/dev/null || echo unknown)
CDATE=$($G log -1 --format=%cI 2>/dev/null || echo unknown)
if $G diff --quiet HEAD -- julia 2>/dev/null; then DIRTY=no; else DIRTY=yes; fi
{
    echo "# The Julia engine in src/ (DFExact.jl, df_series.jl) was copied from DF-Limit by tools/sync-engine.sh"
    echo "upstream_commit: $COMMIT"
    echo "upstream_commit_date: $CDATE"
    echo "upstream_julia_modified: $DIRTY"
    echo "patches:$PATCHES"
} > "$PKG/UPSTREAM"
echo "synced the engine from $UP at $COMMIT (julia/ modified: $DIRTY); patches:${PATCHES:- none}"
