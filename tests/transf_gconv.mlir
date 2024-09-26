// RUN: mlir-opt tests/transf_gconv.mlir --pass-pipeline="builtin.module(transform-interpreter{debug-bind-trailing-args=linalg.generic},canonicalize,cse,symbol-dce)" -o results/tiled_gconv.mlir

!input_tensor_t = tensor<1x128x66x66xf32>
!weight_tensor_t = tensor<256x128x3x3xf32>
!output_tensor_t = tensor<1x256x64x64xf32>

// Função Original a ser otimizada.
// func.func @conv_2d_nchw_fchw(%in: !input_tensor_t, %wei: !weight_tensor_t,
//                              %out: !output_tensor_t) -> !output_tensor_t {
//   %res = linalg.conv_2d_nchw_fchw
//     {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
//      ins(%in, %wei: !input_tensor_t, !weight_tensor_t)
//     outs(%out: !output_tensor_t) -> !output_tensor_t
//   return %res : !output_tensor_t
// }

// linalg-generalize-named-ops:

// d0 = batch; d1 = filtros; d2 = linhas saida; d3 = colunas saida; d4 = canais; d5 = linhas do filtro; d6 = colunas do filtro

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

func.func @conv_2d_nchw_fchw(%in: !input_tensor_t, %wei: !weight_tensor_t,
                               %out: !output_tensor_t) -> !output_tensor_t {
  %0 = linalg.generic
  {indexing_maps = [#map, #map1, #map2],
   iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
   ins(%in, %wei : !input_tensor_t, !weight_tensor_t) outs(%out : !output_tensor_t) {
  ^bb0(%input: f32, %weight: f32, %output: f32):
    %1 = arith.mulf %input, %weight : f32
    %2 = arith.addf %output, %1 : f32
    linalg.yield %2 : f32
  } -> !output_tensor_t
  return %0 : !output_tensor_t
}

module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(
    %arg0: !transform.any_op,
    %conv: !transform.op<"linalg.generic">) {

    transform.debug.emit_remark_at %conv, "Input conv" : !transform.op<"linalg.generic">

    %conv2, %loops2:5 = transform.structured.tile_using_for %conv
      // N, F, OH, OW, C, KH, KW
      tile_sizes [1, 64, 1, 32, 16, 0, 0]   // 16 canais , 32 colunas, 64 filtros, 2o, 1 tile de uma linha
      interchange = [0, 4, 3, 2, 1] // 4 = F, 3: OH, 2:OW, 1:C
      : (!transform.op<"linalg.generic">)
      -> (!transform.op<"linalg.generic">, !transform.any_op, 
          !transform.any_op, !transform.any_op, !transform.any_op, 
          !transform.any_op)

    transform.debug.emit_remark_at %conv2, "conv2" : !transform.op<"linalg.generic">

    %conv3, %loops3:2 = transform.structured.tile_using_for %conv2
      // N, F, OH, OW, C, KH, KW
      tile_sizes [0, 8, 0, 16, 0, 0, 0]
      interchange = [1, 0]
      : (!transform.op<"linalg.generic">)
      -> (!transform.op<"linalg.generic">, !transform.any_op,
          !transform.any_op)

    transform.debug.emit_remark_at %conv3, "conv3" : !transform.op<"linalg.generic">

    // transform.apply_patterns to %conv3 {
    // transform.apply_patterns.canonicalization
    //  transform.apply_patterns.linalg.tiling_canonicalization
    // } : !transform.linalg.generic

    transform.yield
  }

}
