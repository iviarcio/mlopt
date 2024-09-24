// RUN: mlir-opt --lower-affine --cse stride2_aff.mlir -o stride2_loops.mlir-opt

#map = affine_map<(d0, d1) -> (d0 * 2 + d1)>
module {
  func.func @conv_2d_nchw_fchw(%arg0: tensor<1x256x56x56xf32>, %arg1: tensor<128x256x1x1xf32>, %arg2: tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32> {
    %c256 = arith.constant 256 : index
    %c28 = arith.constant 28 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_memref %arg1 : memref<128x256x1x1xf32, strided<[?, ?, ?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg0 : memref<1x256x56x56xf32, strided<[?, ?, ?, ?], offset: ?>>
    %2 = bufferization.to_memref %arg2 : memref<1x128x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x28x28xf32>
    memref.copy %2, %alloc : memref<1x128x28x28xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x128x28x28xf32>
    scf.for %arg3 = %c0 to %c1 step %c1 {
      scf.for %arg4 = %c0 to %c128 step %c1 {
        scf.for %arg5 = %c0 to %c28 step %c1 {
          scf.for %arg6 = %c0 to %c28 step %c1 {
            scf.for %arg7 = %c0 to %c256 step %c1 {
              scf.for %arg8 = %c0 to %c1 step %c1 {
                scf.for %arg9 = %c0 to %c1 step %c1 {
                  %4 = affine.apply #map(%arg5, %arg8)
                  %5 = affine.apply #map(%arg6, %arg9)
                  %6 = memref.load %1[%arg3, %arg7, %4, %5] : memref<1x256x56x56xf32, strided<[?, ?, ?, ?], offset: ?>>
                  %7 = memref.load %0[%arg4, %arg7, %arg8, %arg9] : memref<128x256x1x1xf32, strided<[?, ?, ?, ?], offset: ?>>
                  %8 = memref.load %alloc[%arg3, %arg4, %arg5, %arg6] : memref<1x128x28x28xf32>
                  %9 = arith.mulf %6, %7 : f32
                  %10 = arith.addf %8, %9 : f32
                  memref.store %10, %alloc[%arg3, %arg4, %arg5, %arg6] : memref<1x128x28x28xf32>
                }
              }
            }
          }
        }
      }
    }
    %3 = bufferization.to_tensor %alloc : memref<1x128x28x28xf32>
    return %3 : tensor<1x128x28x28xf32>
  }
}

