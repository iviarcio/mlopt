#!/usr/bin/env bash
#
# Script to apply tile transformations to linalg.generic ops

# Check parameters
if [ $# -lt 1 ]; then
  err "Usage: ./gopt.sh <payload-ir> {without .mlir}"
fi

# Optimize MLIR
mlir-opt ../tests/"$1".mlir --pass-pipeline="builtin.module(transform-interpreter \
    {debug-bind-trailing-args=linalg.generic},cse,symbol-dce)" -o ../results/"$1"_res.mlir
