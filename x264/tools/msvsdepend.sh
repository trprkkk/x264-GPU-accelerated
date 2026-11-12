#!/bin/sh

# Output a Makefile rule describing the dependencies of a given source file.
# Expected arguments are $(CC) $(CFLAGS) $(SRC) $(OBJ)

set -f

[ -n "$1" ] && [ -n "$3" ] && [ -n "$4" ] || exit 1

# Add flags to only perform syntax checking and output a list of included files
# For sources that aren't C, run preprocessing to NUL instead.

case "$3" in
*.c)
    opts="-W0 -Zs"
    ;;
*)
    opts="-P -FiNUL"
    ;;
esac

# The real compile defines the bit depth on its command line
# (-DHIGH_BIT_DEPTH=0 -DBIT_DEPTH=8 for -8.o targets, ...=1 ...=10 for -10.o
# targets).  Without the same defines the syntax check used for dependency
# scanning aborts after too many errors (fatal error C1003) and the include
# list comes out truncated, silently producing incomplete .d files.
# Mirror the defines here, derived from the object file name.
case "$4" in
*-10.o)
    bitdefs="-DHIGH_BIT_DEPTH=1 -DBIT_DEPTH=10"
    ;;
*)
    bitdefs="-DHIGH_BIT_DEPTH=0 -DBIT_DEPTH=8"
    ;;
esac

# Compile to a scratch file so the output can be read twice if needed.
# MSVC localizes the /showIncludes message; on Japanese Windows it prints
# "メモ: インクルード ファイル:  <path>" encoded in the console codepage
# (CP932).  Convert to UTF-8 first so that one pattern matches every locale
# (CP932 and plain ASCII are transparent to iconv).
tmp="$(mktemp 2>/dev/null || echo .x264dep.$$)"
$1 $2 $bitdefs -nologo -showIncludes $opts "$3" > "$tmp" 2>&1
scan="$tmp"
if command -v iconv >/dev/null 2>&1 && iconv -f CP932 -t UTF-8 "$tmp" > "$tmp.u8" 2>/dev/null; then
    scan="$tmp.u8"
fi

# A failed scan would only produce a truncated dependency list; abort loudly
# instead of leaving a silently incomplete .d file behind.
if grep -qE '(fatal )?error [CD][0-9]{4}' "$scan"; then
    echo "msvsdepend.sh: dependency scan of $3 failed:" >&2
    grep -E '(fatal )?error [CD][0-9]{4}' "$scan" | head -5 >&2
    rm -f "$tmp" "$tmp.u8"
    exit 1
fi

# Discard all output other than included files
# Convert '\' directory separators to '/'
# Remove system includes (hack: check for "/Program Files" string in path)
# Add the source file itself as a dependency
deps="$(grep -E '^(Note: including file:|メモ: インクルード ファイル:)' "$scan" |
        sed -E 's/^(Note: including file:|メモ: インクルード ファイル:)[[:space:]]*(.*)$/\2/; s/\\/\//g' |
        sed '/\/[Pp]rogram [Ff]iles/d')
$3"
rm -f "$tmp" "$tmp.u8"

# Convert Windows paths to Unix paths if possible
if command -v cygpath >/dev/null 2>&1 ; then
    IFS='
'
    deps="$(cygpath -u -- $deps)"
elif grep -q 'Microsoft' /proc/sys/kernel/osrelease 2>/dev/null ; then
    # Running under WSL. We don't have access to cygpath but since the Windows
    # file system resides under "/mnt/<drive_letter>/" we can simply replace
    # "C:" with "/mnt/c". This command uses a GNU extension to sed but that's
    # available on WSL so we don't need to limit ourselves by what POSIX says.
    deps="$(printf '%s' "$deps" | sed 's/^\([a-zA-Z]\):/\/mnt\/\L\1/')"
fi

# Native mingw32-make cannot resolve MSYS-style absolute paths ("/q/...",
# "/c/..."), so rewrite paths inside the source tree into the relative form
# that the command line already uses (e.g. "../x264/common/base.h").
# The tree root is located by walking up from the source file's directory
# until x264.h is found, keeping the relative path in lockstep.
src_abs="$(cd "$(dirname "$3")" 2>/dev/null && pwd)"
src_rel="$(dirname "$3")"
while [ -n "$src_abs" ] && [ "$src_abs" != "/" ] && [ ! -f "$src_abs/x264.h" ]; do
    src_abs="$(dirname "$src_abs")"
    src_rel="$(dirname "$src_rel")"
done
if [ -f "$src_abs/x264.h" ]; then
    deps="$(printf '%s' "$deps" | sed "s|^$src_abs/|$src_rel/|")"
fi

# Escape characters as required to create valid Makefile file names
escape() {
    sed 's/ /\\ /g; s/#/\\#/g; s/\$/\$\$/g'
}

# Remove prefixes that are equal to the working directory
# Sort and remove duplicate entries
# Escape and collapse the dependencies into one line
deps="$(printf '%s' "$deps" |
        sed "s/^$(pwd | sed 's/\//\\\//g')\///; s/^\.\///" |
        sort | uniq |
        escape | tr -s '\n\r' ' ' | sed 's/^ *\(.*\) $/\1/')"

# Escape the target file name as well
target="$(printf '%s' "$4" | escape)"

printf '%s: %s\n' "$target" "$deps"
