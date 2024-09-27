// RUN: mlir-opt tests/tile_gsconv.mlir --pass-pipeline="builtin.module(transform-interpreter{debug-bind-trailing-args=linalg.generic},canonicalize,cse,symbol-dce)" -o results/tile_gsconv_res.mlir

// Below, the original gconv.mlir, after applying linalg pass generalize-named-ops on conv.mlir

// definitions:
// d0 = batch; d1 = filtros; d2 = linhas saida; d3 = colunas saida; d4 = canais; d5 = linhas do filtro; d6 = colunas do filtro

// #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
// #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
// #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
// module {
//   func.func @conv_2d_nchw_fchw(%arg0: tensor<1x128x66x66xf32>, %arg1: tensor<256x128x3x3xf32>, %arg2: tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32> {
//     %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x128x66x66xf32>, tensor<256x128x3x3xf32>) outs(%arg2 : tensor<1x256x64x64xf32>) {
//     ^bb0(%in: f32, %in_0: f32, %out: f32):
//       %1 = arith.mulf %in, %in_0 : f32
//       %2 = arith.addf %out, %1 : f32
//       linalg.yield %2 : f32
//     } -> tensor<1x256x64x64xf32>
//     return %0 : tensor<1x256x64x64xf32>
//   }
// }

// The manually modified gconv.mlir to address SConv transformations
// the output, wo, ho (64x64) is linearized to one-dim 4096
// d2 (64) = wo and d3 (64) = ho is modified to d3 (4096) to represent w0xho
// Note the collapse_shape and expand_shape at beginning & end of funtion

#map = affine_map<(d0, d1, d3, d4, d5, d6) -> (d0, d4, d3 floordiv 64 + d5, d3 mod 64 + d6)>
#map1 = affine_map<(d0, d1, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d3, d4, d5, d6) -> (d0, d1, d3)>

module {
  func.func @conv_2d_nchw_fchw(%arg0: tensor<1x128x66x66xf32>, %arg1: tensor<256x128x3x3xf32>, %arg2: tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32> {
    %collapsed = tensor.collapse_shape %arg2 [[0], [1], [2, 3]] : tensor<1x256x64x64xf32> into tensor<1x256x4096xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x128x66x66xf32>, tensor<256x128x3x3xf32>) outs(%collapsed : tensor<1x256x4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<1x256x4096xf32>
    %expanded = tensor.expand_shape %0 [[0], [1], [2, 3]] output_shape [1, 256, 64, 64] : tensor<1x256x4096xf32> into tensor<1x256x64x64xf32>
    return %expanded : tensor<1x256x64x64xf32>
  }
}

module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(
    %arg0: !transform.any_op,
    %conv: !transform.op<"linalg.generic">) {

    transform.debug.emit_remark_at %conv, "Input conv" : !transform.op<"linalg.generic">

    %conv2, %loops2:4 = transform.structured.tile_using_for %conv
      // N, F, OHW, C, KH, KW
      tile_sizes [1, 64, 32, 16, 0, 0]   // 16 C , 32 OHW, 64 F
      interchange = [0, 3, 2, 1] // F, OHW, C
      : (!transform.op<"linalg.generic">)
      -> (!transform.op<"linalg.generic">, !transform.any_op, 
          !transform.any_op, !transform.any_op, !transform.any_op)

    transform.debug.emit_remark_at %conv2, "conv2" : !transform.op<"linalg.generic">

 %conv3, %loops3:2 = transform.structured.tile_using_for %conv2
   // N, F, OHW, C, KH, KW
   tile_sizes [0, 8, 16, 0, 0, 0]
   interchange = [1, 0]
   : (!transform.op<"linalg.generic">)
   -> (!transform.op<"linalg.generic">, !transform.any_op,
       !transform.any_op)

 transform.debug.emit_remark_at %conv3, "conv3" : !transform.op<"linalg.generic">

    transform.apply_patterns to %conv3 {
    transform.apply_patterns.canonicalization
     transform.apply_patterns.linalg.tiling_canonicalization
    } : !transform.op<"linalg.generic">

    transform.yield
  }

}
