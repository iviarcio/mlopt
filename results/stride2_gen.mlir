// RUN: mlir-opt --one-shot-bufferize --convert-linalg-to-loops --cse stride2_gen.mlir -o stride2_aff.mlir-opt

!input_tensor_t = tensor<1x256x56x56xf32>
!weight_tensor_t = tensor<128x256x1x1xf32>
!output_tensor_t = tensor<1x128x28x28xf32>

// Original stride 2 convolution converted into generic.
// func.func @conv_2d_nchw_fchw(%in: !input_tensor_t, %wei: !weight_tensor_t,
//                              %out: !output_tensor_t) -> !output_tensor_t {
//   %res = linalg.conv_2d_nchw_fchw
//     {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
//      ins(%in, %wei: !input_tensor_t, !weight_tensor_t)
//     outs(%out: !output_tensor_t) -> !output_tensor_t
//   return %res : !output_tensor_t
// }

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 * 2 + d5, d3 * 2 + d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

module {
  func.func @conv_2d_nchw_fchw(%arg0: !input_tensor_t, %arg1: !weight_tensor_t, %arg2: !output_tensor_t) -> !output_tensor_t {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : !input_tensor_t, !weight_tensor_t) outs(%arg2 : !output_tensor_t) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> !output_tensor_t
    return %0 : !output_tensor_t
  }
}

