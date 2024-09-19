#!/usr/bin/env bash
#
# Script to apply tile transformations in named linalg ops

# Check parameters
if [ $# -lt 1 ]; then
  err "Usage: ./opt.sh <payload-ir> {without .mlir}"
fi

# Optimize MLIR
mlir-opt ../tests/"$1".mlir --pass-pipeline="builtin.module(transform-interpreter \
    {debug-bind-trailing-args=linalg.conv_2d_nchw_fchw},cse,symbol-dce)" -o ../results/"$1"_res.mlir
