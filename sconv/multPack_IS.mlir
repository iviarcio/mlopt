#map = affine_map<(d0) -> (d0 floordiv 64)>
#map1 = affine_map<(d0) -> (d0 mod 64)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0) -> (d0 floordiv 9)>
#map4 = affine_map<(d0) -> (d0 mod 9)>
#map5 = affine_map<(d0) -> ((d0 mod 9) floordiv 3)>
#map6 = affine_map<(d0) -> (d0 mod 3)>
#map7 = affine_map<(d0, d1) -> (d0 * 8 + d1)>
#map8 = affine_map<(d0, d1) -> (d0 + d1 - (d1 floordiv 3) * 3)>
#map9 = affine_map<(d0) -> (d0 floordiv 8)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d3, d1)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
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
          %6 = tensor.empty() : tensor<8x144x8xf32>
          %7 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%6 : tensor<8x144x8xf32>) {
          ^bb0(%out: f32):
            %9 = linalg.index 0 : index
            %10 = linalg.index 1 : index
            %11 = linalg.index 2 : index
            %c16_11 = arith.constant 16 : index
            %c3 = arith.constant 3 : index
            %c3_12 = arith.constant 3 : index
            %c3_13 = arith.constant 3 : index
            %c9 = arith.constant 9 : index
            %12 = affine.apply #map3(%10)
            %13 = affine.apply #map4(%10)
            %14 = affine.apply #map5(%10)
            %15 = affine.apply #map6(%10)
            %16 = affine.apply #map7(%9, %11)
            %extracted = tensor.extract %extracted_slice_4[%16, %12, %14, %15] : tensor<64x16x3x3xf32>
            linalg.yield %extracted : f32
          } -> tensor<8x144x8xf32>
          %8 = scf.for %arg11 = %c0_7 to %c32_9 step %c16_10 iter_args(%arg12 = %extracted_slice_5) -> (tensor<1x64x32xf32>) {
            %9 = affine.apply #map(%arg11)
            %10 = affine.apply #map1(%arg11)
            %extracted_slice_11 = tensor.extract_slice %extracted_slice[0, 0, %9, %10] [1, 16, 3, 18] [1, 1, 1, 1] : tensor<1x16x3x34xf32> to tensor<1x16x3x18xf32>
            %11 = tensor.empty() : tensor<1x144x16xf32>
            %12 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%11 : tensor<1x144x16xf32>) {
            ^bb0(%out: f32):
              %14 = linalg.index 0 : index
              %15 = linalg.index 1 : index
              %16 = linalg.index 2 : index
              %c16_12 = arith.constant 16 : index
              %c3 = arith.constant 3 : index
              %c3_13 = arith.constant 3 : index
              %c3_14 = arith.constant 3 : index
              %c9 = arith.constant 9 : index
              %17 = affine.apply #map3(%15)
              %18 = affine.apply #map4(%15)
              %19 = affine.apply #map5(%15)
              %20 = affine.apply #map6(%15)
              %21 = affine.apply #map8(%16, %15)
              %extracted = tensor.extract %extracted_slice_11[%14, %17, %19, %21] : tensor<1x16x3x18xf32>
              linalg.yield %extracted : f32
            } -> tensor<1x144x16xf32>
            %13 = scf.for %arg13 = %c0_6 to %c64_8 step %c8 iter_args(%arg14 = %arg12) -> (tensor<1x64x32xf32>) {
              %14 = affine.apply #map9(%arg13)
              %extracted_slice_12 = tensor.extract_slice %7[%14, 0, 0] [1, 144, 8] [1, 1, 1] : tensor<8x144x8xf32> to tensor<1x144x8xf32>
              %collapsed_13 = tensor.collapse_shape %extracted_slice_12 [[0, 1], [2]] : tensor<1x144x8xf32> into tensor<144x8xf32>
              %extracted_slice_14 = tensor.extract_slice %arg14[0, %arg13, %arg11] [1, 8, 16] [1, 1, 1] : tensor<1x64x32xf32> to tensor<1x8x16xf32>
              %15 = linalg.generic {indexing_maps = [#map10, #map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%12, %collapsed_13 : tensor<1x144x16xf32>, tensor<144x8xf32>) outs(%extracted_slice_14 : tensor<1x8x16xf32>) {
              ^bb0(%in: f32, %in_16: f32, %out: f32):
                %16 = arith.mulf %in, %in_16 : f32
                %17 = arith.addf %16, %out : f32
                linalg.yield %17 : f32
              } -> tensor<1x8x16xf32>
              %inserted_slice_15 = tensor.insert_slice %15 into %arg14[0, %arg13, %arg11] [1, 8, 16] [1, 1, 1] : tensor<1x8x16xf32> into tensor<1x64x32xf32>
              scf.yield %inserted_slice_15 : tensor<1x64x32xf32>
            }
            scf.yield %13 : tensor<1x64x32xf32>
          }
          %inserted_slice = tensor.insert_slice %8 into %arg10[%arg3, %arg9, %arg7] [1, 64, 32] [1, 1, 1] : tensor<1x64x32xf32> into tensor<1x256x4096xf32>
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
