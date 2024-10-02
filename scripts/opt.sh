#!/usr/bin/env bash

# Script to apply transformations to MLIR files
# Make sure to set the LLVM_BIN_PATH variable to the LLVM build bin path

# old-version: Optimize MLIR
# mlir-opt ../tests/"$1".mlir --pass-pipeline="builtin.module(transform-interpreter \
#     {debug-bind-trailing-args=linalg.conv_2d_nchw_fchw},cse,symbol-dce)" -o ../results/"$1"_res.mlir

err() {
  echo "$*" >&2
  exit 1
}

# Check parameters
if [ $# -lt 2 ]; then
  err "Usage: ./opt.sh <payload-ir>.mlir <transform-ir>.mlir"
fi

# Create output file name
INPUT_BASE=$(basename "$1" .mlir)
readonly INPUT_BASE

TRANSFORM_BASE=$(basename "$2" .mlir)
readonly TRANSFORM_BASE

# Check if input and transform are .mlir files
if [[ "$INPUT_BASE" == "$1" || "$TRANSFORM_BASE" == "$2" ]]; then
  err "Files needs to be .mlir"
fi

# Create output file name
INPUT_DIR=$(dirname "$1")
readonly INPUT_DIR
readonly OUTPUT="$INPUT_DIR"/"$INPUT_BASE".opt.mlir

# Optimize MLIR
mlir-transform-opt -transform="$2" "$1" > "$OUTPUT"
