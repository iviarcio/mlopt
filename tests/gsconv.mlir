// The original gconv.mlir, applying linalg.generalize-named-ops on conv.mlir

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

// The modified gconv.mlir that's calling gsconv.mlir to address SConv transformations
// the output, wo, ho (64x64) is linearized to one-dim 4096
// d2 (64) = wo and d3 (64) = ho is modified to d3 (4096) to represent w0xho

#map = affine_map<(d0, d1, d3, d4, d5, d6) -> (d0, d4, d3 floordiv 64 + d5, d3 mod 64 + d6)>
#map1 = affine_map<(d0, d1, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d3, d4, d5, d6) -> (d0, d1, d3)>
module {
  func.func @conv_2d_nchw_fchw(%arg0: tensor<1x128x66x66xf32>, %arg1: tensor<256x128x3x3xf32>, %arg2: tensor<1x256x4096xf32>) -> tensor<1x256x4096xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x128x66x66xf32>, tensor<256x128x3x3xf32>) outs(%arg2 : tensor<1x256x4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<1x256x4096xf32>
    return %0 : tensor<1x256x4096xf32>
  }
}

