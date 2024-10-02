#!/usr/bin/env bash

# Script to test a transform sequence on a MLIR payload file.
# Tries to lower the payload directly and then with the transform sequence.
# Compares the outputs.
# The test to be run is specified in the payload file.
# Uses opt_and_run.sh and run.sh scripts.

# Important: ignores the first line of the output.

err() { echo "$*" >&2; exit 1; }

# Check parameters
if [ $# -lt 2 ]; then
  err "Usage: ./test_transform.sh <payload-ir>.mlir <transform-ir>.mlir [-O]"
fi
readonly PAYLOAD=$1
readonly TRANSFORM=$2
shift 2
OPTIMIZATION=""
while getopts 'O' flag; do
  case "${flag}" in
  O) OPTIMIZATION="-O" ;;
  *) error "Unexpected option ${flag}" ;;
  esac
done
readonly OPTIMIZATION

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
readonly SCRIPT

# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")
readonly SCRIPTPATH

# JIT compile and run MLIR payload directly
CORRECT_OUT=$("$SCRIPTPATH"/run.sh "$PAYLOAD" "$OPTIMIZATION")
readonly CORRECT_OUT

# Transform and JIT compile and run MLIR payload
OUT=$("$SCRIPTPATH"/opt_and_run.sh "$PAYLOAD" "$TRANSFORM" "$OPTIMIZATION")
readonly OUT

# Compare outputs, ignoring the first line
# (this should be considered in the main function)
DIFF=$(
  diff \
    <(printf '%s\n' "$CORRECT_OUT" | tail -1) \
    <(printf '%s\n' "$OUT" | tail -1)
)
readonly DIFF

# Check if diff was successful
if [[ "$DIFF" != "" ]]; then
  echo "FAILURE"
else
  echo "SUCCESS"
fi
