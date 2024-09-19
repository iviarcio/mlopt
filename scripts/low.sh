#!/usr/bin/env bash
#
# Script to apply lower pass in linalg generic ops

# Check parameters
if [ $# -lt 1 ]; then
  err "Usage: ./opt.sh <payload-ir> {without .mlir}"
fi

# Lower linalg to loops
mlir-opt --one-shot-bufferize --convert-linalg-to-loops --lower-affine --cse ../tests/"$1".mlir -o ../results/"$1"_loops.mlir

mlir-opt ../tests/"$1".mlir --pass-pipeline="builtin.module(transform-interpreter \
    {debug-bind-trailing-args=linalg.conv_2d_nchw_fchw},cse,symbol-dce)" -o ../results/"$1"_res.mlir
