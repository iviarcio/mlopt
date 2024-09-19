module {
  func.func @conv_2d_nchw_fchw(%arg0: tensor<1x128x66x66xf32>, %arg1: tensor<256x128x3x3xf32>, %arg2: tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32> {
    %c3 = arith.constant 3 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_memref %arg1 : memref<256x128x3x3xf32, strided<[?, ?, ?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg0 : memref<1x128x66x66xf32, strided<[?, ?, ?, ?], offset: ?>>
    %2 = bufferization.to_memref %arg2 : memref<1x256x64x64xf32, strided<[?, ?, ?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x256x64x64xf32>
    memref.copy %2, %alloc : memref<1x256x64x64xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x256x64x64xf32>
    scf.for %arg3 = %c0 to %c1 step %c1 {
      scf.for %arg4 = %c0 to %c256 step %c1 {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          scf.for %arg6 = %c0 to %c64 step %c1 {
            scf.for %arg7 = %c0 to %c128 step %c1 {
              scf.for %arg8 = %c0 to %c3 step %c1 {
                scf.for %arg9 = %c0 to %c3 step %c1 {
                  %4 = arith.addi %arg5, %arg8 : index
                  %5 = arith.addi %arg6, %arg9 : index
                  %6 = memref.load %1[%arg3, %arg7, %4, %5] : memref<1x128x66x66xf32, strided<[?, ?, ?, ?], offset: ?>>
                  %7 = memref.load %0[%arg4, %arg7, %arg8, %arg9] : memref<256x128x3x3xf32, strided<[?, ?, ?, ?], offset: ?>>
                  %8 = memref.load %alloc[%arg3, %arg4, %arg5, %arg6] : memref<1x256x64x64xf32>
                  %9 = arith.mulf %6, %7 : f32
                  %10 = arith.addf %8, %9 : f32
                  memref.store %10, %alloc[%arg3, %arg4, %arg5, %arg6] : memref<1x256x64x64xf32>
                }
              }
            }
          }
        }
      }
    }
    %3 = bufferization.to_tensor %alloc : memref<1x256x64x64xf32>
    return %3 : tensor<1x256x64x64xf32>
  }
}

