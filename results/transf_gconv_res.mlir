#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
module {
  func.func @conv_2d_nchw_fchw(%arg0: tensor<1x128x66x66xf32>, %arg1: tensor<256x128x3x3xf32>, %arg2: tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %0 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %arg2) -> (tensor<1x256x64x64xf32>) {
      %1 = scf.for %arg5 = %c0 to %c128 step %c16 iter_args(%arg6 = %arg4) -> (tensor<1x256x64x64xf32>) {
        %2 = scf.for %arg7 = %c0 to %c64 step %c32 iter_args(%arg8 = %arg6) -> (tensor<1x256x64x64xf32>) {
          %3 = scf.for %arg9 = %c0 to %c64 step %c1 iter_args(%arg10 = %arg8) -> (tensor<1x256x64x64xf32>) {
            %4 = scf.for %arg11 = %c0 to %c256 step %c64 iter_args(%arg12 = %arg10) -> (tensor<1x256x64x64xf32>) {
              %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg5, %arg9, %arg7] [1, 16, 3, 34] [1, 1, 1, 1] : tensor<1x128x66x66xf32> to tensor<1x16x3x34xf32>
              %extracted_slice_0 = tensor.extract_slice %arg1[%arg11, %arg5, 0, 0] [64, 16, 3, 3] [1, 1, 1, 1] : tensor<256x128x3x3xf32> to tensor<64x16x3x3xf32>
              %extracted_slice_1 = tensor.extract_slice %arg12[%arg3, %arg11, %arg9, %arg7] [1, 64, 1, 32] [1, 1, 1, 1] : tensor<1x256x64x64xf32> to tensor<1x64x1x32xf32>
              %c8 = arith.constant 8 : index
              %5 = scf.for %arg13 = %c0 to %c64 step %c8 iter_args(%arg14 = %extracted_slice_1) -> (tensor<1x64x1x32xf32>) {
                %6 = scf.for %arg15 = %c0 to %c32 step %c16 iter_args(%arg16 = %arg14) -> (tensor<1x64x1x32xf32>) {
                  %extracted_slice_2 = tensor.extract_slice %extracted_slice[0, 0, 0, %arg15] [1, 16, 3, 18] [1, 1, 1, 1] : tensor<1x16x3x34xf32> to tensor<1x16x3x18xf32>
                  %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[%arg13, 0, 0, 0] [8, 16, 3, 3] [1, 1, 1, 1] : tensor<64x16x3x3xf32> to tensor<8x16x3x3xf32>
                  %extracted_slice_4 = tensor.extract_slice %arg16[0, %arg13, 0, %arg15] [1, 8, 1, 16] [1, 1, 1, 1] : tensor<1x64x1x32xf32> to tensor<1x8x1x16xf32>
                  %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice_2, %extracted_slice_3 : tensor<1x16x3x18xf32>, tensor<8x16x3x3xf32>) outs(%extracted_slice_4 : tensor<1x8x1x16xf32>) {
                  ^bb0(%in: f32, %in_6: f32, %out: f32):
                    %8 = arith.mulf %in, %in_6 : f32
                    %9 = arith.addf %out, %8 : f32
                    linalg.yield %9 : f32
                  } -> tensor<1x8x1x16xf32>
                  %inserted_slice_5 = tensor.insert_slice %7 into %arg16[0, %arg13, 0, %arg15] [1, 8, 1, 16] [1, 1, 1, 1] : tensor<1x8x1x16xf32> into tensor<1x64x1x32xf32>
                  scf.yield %inserted_slice_5 : tensor<1x64x1x32xf32>
                }
                scf.yield %6 : tensor<1x64x1x32xf32>
              }
              %inserted_slice = tensor.insert_slice %5 into %arg12[%arg3, %arg11, %arg9, %arg7] [1, 64, 1, 32] [1, 1, 1, 1] : tensor<1x64x1x32xf32> into tensor<1x256x64x64xf32>
              scf.yield %inserted_slice : tensor<1x256x64x64xf32>
            }
            scf.yield %4 : tensor<1x256x64x64xf32>
          }
          scf.yield %3 : tensor<1x256x64x64xf32>
        }
        scf.yield %2 : tensor<1x256x64x64xf32>
      }
      scf.yield %1 : tensor<1x256x64x64xf32>
    }
    return %0 : tensor<1x256x64x64xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op, %arg1: !transform.op<"linalg.generic">) {
      %tiled_linalg_op, %loops:5 = transform.structured.tile_using_for %arg1 tile_sizes [1, 64, 1, 32, 16, 0, 0] interchange = [0, 4, 3, 2, 1] : (!transform.op<"linalg.generic">) -> (!transform.op<"linalg.generic">, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      %tiled_linalg_op_0, %loops_1:2 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 8, 0, 16, 0, 0, 0] interchange = [1, 0] : (!transform.op<"linalg.generic">) -> (!transform.op<"linalg.generic">, !transform.any_op, !transform.any_op)
      transform.yield 
    }
  }
}

