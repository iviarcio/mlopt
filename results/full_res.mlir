#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0) -> (d0 floordiv 9)>
#map2 = affine_map<(d0, d1) -> (d0 floordiv 16 + (d1 mod 9) floordiv 3)>
#map3 = affine_map<(d0, d1) -> (d0 + d1 - (d0 floordiv 16) * 16 - (d1 floordiv 3) * 3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
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
                  %collapsed = tensor.collapse_shape %extracted_slice_3 [[0], [1, 2, 3]] : tensor<8x16x3x3xf32> into tensor<8x144xf32>
                  %collapsed_5 = tensor.collapse_shape %extracted_slice_4 [[0], [1], [2, 3]] : tensor<1x8x1x16xf32> into tensor<1x8x16xf32>
                  %7 = tensor.empty() : tensor<1x144x16xf32>
                  %8 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%7 : tensor<1x144x16xf32>) {
                  ^bb0(%out: f32):
                    %10 = linalg.index 0 : index
                    %11 = linalg.index 1 : index
                    %12 = linalg.index 2 : index
                    %13 = affine.apply #map1(%11)
                    %14 = affine.apply #map2(%12, %11)
                    %15 = affine.apply #map3(%12, %11)
                    %extracted = tensor.extract %extracted_slice_2[%10, %13, %14, %15] : tensor<1x16x3x18xf32>
                    linalg.yield %extracted : f32
                  } -> tensor<1x144x16xf32>
                  %9 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed, %8 : tensor<8x144xf32>, tensor<1x144x16xf32>) outs(%collapsed_5 : tensor<1x8x16xf32>) {
                  ^bb0(%in: f32, %in_7: f32, %out: f32):
                    %10 = arith.mulf %in, %in_7 : f32
                    %11 = arith.addf %10, %out : f32
                    linalg.yield %11 : f32
                  } -> tensor<1x8x16xf32>
                  %expanded = tensor.expand_shape %9 [[0], [1], [2, 3]] output_shape [1, 8, 1, 16] : tensor<1x8x16xf32> into tensor<1x8x1x16xf32>
                  %inserted_slice_6 = tensor.insert_slice %expanded into %arg16[0, %arg13, 0, %arg15] [1, 8, 1, 16] [1, 1, 1, 1] : tensor<1x8x1x16xf32> into tensor<1x64x1x32xf32>
                  scf.yield %inserted_slice_6 : tensor<1x64x1x32xf32>
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
    transform.named_sequence @__transform_main(%arg0: !transform.any_op, %arg1: !transform.op<"linalg.conv_2d_nchw_fchw">) {
      %tiled_linalg_op, %loops:5 = transform.structured.tile_using_for %arg1 tile_sizes [1, 64, 1, 32, 16, 0, 0] interchange = [0, 4, 3, 2, 1] : (!transform.op<"linalg.conv_2d_nchw_fchw">) -> (!transform.op<"linalg.conv_2d_nchw_fchw">, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      %tiled_linalg_op_0, %loops_1:2 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 8, 0, 16, 0, 0, 0] interchange = [1, 0] : (!transform.op<"linalg.conv_2d_nchw_fchw">) -> (!transform.op<"linalg.conv_2d_nchw_fchw">, !transform.any_op, !transform.any_op)
      %img2col_tensor, %transformed = transform.structured.convert_conv2d_to_img2col %tiled_linalg_op_0 : (!transform.op<"linalg.conv_2d_nchw_fchw">) -> (!transform.any_op, !transform.any_op)
      transform.apply_patterns to %img2col_tensor {
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
      transform.yield 
    }
  }
}

