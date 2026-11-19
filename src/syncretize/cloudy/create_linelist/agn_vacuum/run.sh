#!/bin/sh
# Run one Cloudy model of this directory: sh run.sh <name> reads <name>.in
# and writes <name>.out (and the files the input saves).
#
# CLOUDY_EXE      path of the cloudy.exe to use (required).
# CLOUDY_DATA_PATH  Cloudy's data directory; by default the data/ directory
#                 next to the build's source/. ./ is appended so that local
#                 files (e.g. the .sed) are found.
# CLOUDY_EXTRA    optional commands prepended to the input, e.g.
#                 CLOUDY_EXTRA="set continuum resolution 0.05" for a build
#                 whose grain opacities were compiled at that resolution.
: "${CLOUDY_EXE:?set CLOUDY_EXE to the cloudy.exe to use}"
CLOUDY_DATA_PATH="${CLOUDY_DATA_PATH:-$(dirname "$CLOUDY_EXE")/../data/}:./"
export CLOUDY_DATA_PATH
{
    [ -n "$CLOUDY_EXTRA" ] && printf '%s\n' "$CLOUDY_EXTRA"
    cat "$1.in"
    echo
} | "$CLOUDY_EXE" > "$1.out"
