#!/bin/sh
# Run every model (*.in) of this directory with run.sh (see run.sh for the
# environment variables, e.g. CLOUDY_EXE).
for input in *.in; do
    sh run.sh "${input%.in}"
done
