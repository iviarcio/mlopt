module {
  func.func @conv_2d_nchw_fchw(%arg0: tensor<1x128x66x66xf32>, %arg1: tensor<256x128x3x3xf32>, %arg2: tensor<1x256x4096xf32>) -> tensor<1x256x4096xf32> {
    %c3 = arith.constant 3 : index
    %c128 = arith.constant 128 : index
    %c4096 = arith.constant 4096 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_memref %arg1 : memref<256x128x3x3xf32, strided<[?, ?, ?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg0 : memref<1x128x66x66xf32, strided<[?, ?, ?, ?], offset: ?>>
    %2 = bufferization.to_memref %arg2 : memref<1x256x4096xf32, strided<[?, ?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x256x4096xf32>
    memref.copy %2, %alloc : memref<1x256x4096xf32, strided<[?, ?, ?], offset: ?>> to memref<1x256x4096xf32>
    scf.for %arg3 = %c0 to %c1 step %c1 {
      scf.for %arg4 = %c0 to %c256 step %c1 {
        scf.for %arg5 = %c0 to %c4096 step %c1 {
          scf.for %arg6 = %c0 to %c128 step %c1 {
            scf.for %arg7 = %c0 to %c3 step %c1 {
              scf.for %arg8 = %c0 to %c3 step %c1 {
                %c64 = arith.constant 64 : index
                %c-1 = arith.constant -1 : index
                %4 = arith.cmpi slt, %arg5, %c0 : index
                %5 = arith.subi %c-1, %arg5 : index
                %6 = arith.select %4, %5, %arg5 : index
                %7 = arith.divsi %6, %c64 : index
                %8 = arith.subi %c-1, %7 : index
                %9 = arith.select %4, %8, %7 : index
                %10 = arith.addi %arg7, %9 : index
                %11 = arith.addi %arg5, %arg8 : index
                %c-64 = arith.constant -64 : index
                %12 = arith.muli %9, %c-64 : index
                %13 = arith.addi %11, %12 : index
                %14 = memref.load %1[%arg3, %arg6, %10, %13] : memref<1x128x66x66xf32, strided<[?, ?, ?, ?], offset: ?>>
                %15 = memref.load %0[%arg4, %arg6, %arg7, %arg8] : memref<256x128x3x3xf32, strided<[?, ?, ?, ?], offset: ?>>
                %16 = memref.load %alloc[%arg3, %arg4, %arg5] : memref<1x256x4096xf32>
                %17 = arith.mulf %14, %15 : f32
                %18 = arith.addf %16, %17 : f32
                memref.store %18, %alloc[%arg3, %arg4, %arg5] : memref<1x256x4096xf32>
              }
            }
          }
        }
      }
    }
    %3 = bufferization.to_tensor %alloc : memref<1x256x4096xf32>
    return %3 : tensor<1x256x4096xf32>
  }
}

