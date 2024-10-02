#map = affine_map<(d0) -> (d0 floordiv 64)>
#map1 = affine_map<(d0) -> (d0 mod 64)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d2 floordiv 64 + d4, d2 mod 64 + d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d4, d5)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
module {
  func.func @conv_2d_nchw_fchw(%arg0: tensor<1x128x66x66xf32>, %arg1: tensor<256x128x3x3xf32>, %arg2: tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32> {
    %collapsed = tensor.collapse_shape %arg2 [[0], [1], [2, 3]] : tensor<1x256x64x64xf32> into tensor<1x256x4096xf32>
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %c0_2 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c4096 = arith.constant 4096 : index
    %c256 = arith.constant 256 : index
    %c1_3 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %0 = scf.for %arg3 = %c0 to %c1 step %c1_3 iter_args(%arg4 = %collapsed) -> (tensor<1x256x4096xf32>) {
      %1 = scf.for %arg5 = %c0_0 to %c128 step %c16 iter_args(%arg6 = %arg4) -> (tensor<1x256x4096xf32>) {
        %2 = scf.for %arg7 = %c0_1 to %c4096 step %c32 iter_args(%arg8 = %arg6) -> (tensor<1x256x4096xf32>) {
          %3 = scf.for %arg9 = %c0_2 to %c256 step %c64 iter_args(%arg10 = %arg8) -> (tensor<1x256x4096xf32>) {
            %4 = affine.apply #map(%arg7)
            %5 = affine.apply #map1(%arg7)
            %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg5, %4, %5] [1, 16, 3, 34] [1, 1, 1, 1] : tensor<1x128x66x66xf32> to tensor<1x16x3x34xf32>
            %extracted_slice_4 = tensor.extract_slice %arg1[%arg9, %arg5, 0, 0] [64, 16, 3, 3] [1, 1, 1, 1] : tensor<256x128x3x3xf32> to tensor<64x16x3x3xf32>
            %extracted_slice_5 = tensor.extract_slice %arg10[%arg3, %arg9, %arg7] [1, 64, 32] [1, 1, 1] : tensor<1x256x4096xf32> to tensor<1x64x32xf32>
            %c0_6 = arith.constant 0 : index
            %c0_7 = arith.constant 0 : index
            %c64_8 = arith.constant 64 : index
            %c32_9 = arith.constant 32 : index
            %c8 = arith.constant 8 : index
            %c16_10 = arith.constant 16 : index
            %6 = scf.for %arg11 = %c0_6 to %c64_8 step %c8 iter_args(%arg12 = %extracted_slice_5) -> (tensor<1x64x32xf32>) {
              %7 = scf.for %arg13 = %c0_7 to %c32_9 step %c16_10 iter_args(%arg14 = %arg12) -> (tensor<1x64x32xf32>) {
                %8 = affine.apply #map(%arg13)
                %9 = affine.apply #map1(%arg13)
                %extracted_slice_11 = tensor.extract_slice %extracted_slice[0, 0, %8, %9] [1, 16, 3, 18] [1, 1, 1, 1] : tensor<1x16x3x34xf32> to tensor<1x16x3x18xf32>
                %extracted_slice_12 = tensor.extract_slice %extracted_slice_4[%arg11, 0, 0, 0] [8, 16, 3, 3] [1, 1, 1, 1] : tensor<64x16x3x3xf32> to tensor<8x16x3x3xf32>
                %extracted_slice_13 = tensor.extract_slice %arg14[0, %arg11, %arg13] [1, 8, 16] [1, 1, 1] : tensor<1x64x32xf32> to tensor<1x8x16xf32>
                %10 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_11, %extracted_slice_12 : tensor<1x16x3x18xf32>, tensor<8x16x3x3xf32>) outs(%extracted_slice_13 : tensor<1x8x16xf32>) {
                ^bb0(%in: f32, %in_15: f32, %out: f32):
                  %11 = arith.mulf %in, %in_15 : f32
                  %12 = arith.addf %out, %11 : f32
                  linalg.yield %12 : f32
                } -> tensor<1x8x16xf32>
                %inserted_slice_14 = tensor.insert_slice %10 into %arg14[0, %arg11, %arg13] [1, 8, 16] [1, 1, 1] : tensor<1x8x16xf32> into tensor<1x64x32xf32>
                scf.yield %inserted_slice_14 : tensor<1x64x32xf32>
              }
              scf.yield %7 : tensor<1x64x32xf32>
            }
            %inserted_slice = tensor.insert_slice %6 into %arg10[%arg3, %arg9, %arg7] [1, 64, 32] [1, 1, 1] : tensor<1x64x32xf32> into tensor<1x256x4096xf32>
            scf.yield %inserted_slice : tensor<1x256x4096xf32>
          }
          scf.yield %3 : tensor<1x256x4096xf32>
        }
        scf.yield %2 : tensor<1x256x4096xf32>
      }
      scf.yield %1 : tensor<1x256x4096xf32>
    }
    %expanded = tensor.expand_shape %0 [[0], [1], [2, 3]] output_shape [1, 256, 64, 64] : tensor<1x256x4096xf32> into tensor<1x256x64x64xf32>
    return %expanded : tensor<1x256x64x64xf32>
  }
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %generated = tensor.generate  {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      %8 = arith.addi %arg3, %arg2 : index
      %9 = arith.addi %8, %arg1 : index
      %10 = arith.addi %9, %arg0 : index
      %11 = arith.index_cast %10 : index to i64
      %12 = arith.uitofp %11 : i64 to f32
      tensor.yield %12 : f32
    } : tensor<1x128x66x66xf32>
    %generated_0 = tensor.generate  {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      %8 = arith.addi %arg3, %arg2 : index
      %9 = arith.addi %8, %arg1 : index
      %10 = arith.addi %9, %arg0 : index
      %11 = arith.index_cast %10 : index to i64
      %12 = arith.uitofp %11 : i64 to f32
      tensor.yield %12 : f32
    } : tensor<256x128x3x3xf32>
    %generated_1 = tensor.generate  {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      %8 = arith.addi %arg3, %arg2 : index
      %9 = arith.addi %8, %arg1 : index
      %10 = arith.addi %9, %arg0 : index
      %11 = arith.index_cast %10 : index to i64
      %12 = arith.uitofp %11 : i64 to f32
      tensor.yield %12 : f32
    } : tensor<1x256x64x64xf32>
    %c1_2 = arith.constant 1 : index
    %0 = call @rtclock() : () -> f64
    %1 = scf.for %arg0 = %c0 to %c1_2 step %c1 iter_args(%arg1 = %generated_1) -> (tensor<1x256x64x64xf32>) {
      %8 = func.call @conv_2d_nchw_fchw(%generated, %generated_0, %arg1) : (tensor<1x128x66x66xf32>, tensor<256x128x3x3xf32>, tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
      scf.yield %8 : tensor<1x256x64x64xf32>
    }
    %2 = call @rtclock() : () -> f64
    %3 = arith.subf %2, %0 : f64
    %cast = tensor.cast %1 : tensor<1x256x64x64xf32> to tensor<*xf32>
    call @printMemrefF32(%cast) : (tensor<*xf32>) -> ()
    %c2415919104 = arith.constant 2415919104 : index
    %4 = arith.muli %c2415919104, %c1_2 : index
    %5 = arith.index_cast %4 : index to i64
    %6 = arith.uitofp %5 : i64 to f64
    %7 = arith.divf %6, %3 : f64
    call @printFlops(%7) : (f64) -> ()
    return
  }
  func.func private @printMemrefF32(tensor<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @rtclock() -> f64
}
