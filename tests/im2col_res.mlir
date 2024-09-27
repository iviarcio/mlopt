#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0) -> (d0 floordiv 9)>
#map2 = affine_map<(d0, d1) -> (d0 floordiv 64 + (d1 mod 9) floordiv 3)>
#map3 = affine_map<(d0, d1) -> (d0 + d1 - (d0 floordiv 64) * 64 - (d1 floordiv 3) * 3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @conv_2d_nchw_fchw(%arg0: tensor<1x128x66x66xf32>, %arg1: tensor<256x128x3x3xf32>, %arg2: tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32> {
    %collapsed = tensor.collapse_shape %arg1 [[0], [1, 2, 3]] : tensor<256x128x3x3xf32> into tensor<256x1152xf32>
    %collapsed_0 = tensor.collapse_shape %arg2 [[0], [1], [2, 3]] : tensor<1x256x64x64xf32> into tensor<1x256x4096xf32>
    %0 = tensor.empty() : tensor<1x1152x4096xf32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%0 : tensor<1x1152x4096xf32>) {
    ^bb0(%out: f32):
      %3 = linalg.index 0 : index
      %4 = linalg.index 1 : index
      %5 = linalg.index 2 : index
      %6 = affine.apply #map1(%4)
      %7 = affine.apply #map2(%5, %4)
      %8 = affine.apply #map3(%5, %4)
      %extracted = tensor.extract %arg0[%3, %6, %7, %8] : tensor<1x128x66x66xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x1152x4096xf32>
    %2 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed, %1 : tensor<256x1152xf32>, tensor<1x1152x4096xf32>) outs(%collapsed_0 : tensor<1x256x4096xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %3 = arith.mulf %in, %in_1 : f32
      %4 = arith.addf %3, %out : f32
      linalg.yield %4 : f32
    } -> tensor<1x256x4096xf32>
    %expanded = tensor.expand_shape %2 [[0], [1], [2, 3]] output_shape [1, 256, 64, 64] : tensor<1x256x4096xf32> into tensor<1x256x64x64xf32>
    return %expanded : tensor<1x256x64x64xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op, %arg1: !transform.op<"linalg.conv_2d_nchw_fchw">) {
      %img2col_tensor, %transformed = transform.structured.convert_conv2d_to_img2col %arg1 : (!transform.op<"linalg.conv_2d_nchw_fchw">) -> (!transform.any_op, !transform.any_op)
      transform.apply_patterns to %img2col_tensor {
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
      transform.yield 
    }
  }
}

