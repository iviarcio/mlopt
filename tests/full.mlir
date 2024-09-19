// RUN: mlir-opt "$1".mlir --pass-pipeline=" \
// RUN:   builtin.module(transform-interpreter{ \
// RUN:     debug-bind-trailing-args=linalg.conv_2d_nchw_fchw},canonicalize,cse,symbol-dce)" \
// RUN:   -o "$1"_res.mlir

!input_tensor_t = tensor<1x128x66x66xf32>
!weight_tensor_t = tensor<256x128x3x3xf32>
!output_tensor_t = tensor<1x256x64x64xf32>

// Função Original a ser otimizada.
func.func @conv_2d_nchw_fchw(%in: !input_tensor_t, %wei: !weight_tensor_t,
                             %out: !output_tensor_t) -> !output_tensor_t {
  %res = linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%in, %wei: !input_tensor_t, !weight_tensor_t)
    outs(%out: !output_tensor_t) -> !output_tensor_t
  return %res : !output_tensor_t
}

module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(
    %arg0: !transform.any_op,
    %conv: !transform.op<"linalg.conv_2d_nchw_fchw">) {

    transform.debug.emit_remark_at %conv, "Input conv" : !transform.op<"linalg.conv_2d_nchw_fchw">

    %conv2, %loops2:5 = transform.structured.tile_using_for %conv
      // N, F, OH, OW, C, KH, KW
      tile_sizes [1, 64, 1, 32, 16, 0, 0]   // 16 canais , 32 colunas, 64 filtros, 2o, 1 tile de uma linha
      interchange = [0, 4, 3, 2, 1] // 4 = F, 3: OH, 2:OW, 1:C
      : (!transform.op<"linalg.conv_2d_nchw_fchw">)
      -> (!transform.op<"linalg.conv_2d_nchw_fchw">, !transform.any_op, 
          !transform.any_op, !transform.any_op, !transform.any_op, 
          !transform.any_op)

    transform.debug.emit_remark_at %conv2, "conv2" : !transform.op<"linalg.conv_2d_nchw_fchw">

    %conv3, %loops3:2 = transform.structured.tile_using_for %conv2
      // N, F, OH, OW, C, KH, KW
      tile_sizes [0, 8, 0, 16, 0, 0, 0]
      interchange = [1, 0]
      : (!transform.op<"linalg.conv_2d_nchw_fchw">)
      -> (!transform.op<"linalg.conv_2d_nchw_fchw">, !transform.any_op,
          !transform.any_op)

    transform.debug.emit_remark_at %conv3, "conv3" : !transform.op<"linalg.conv_2d_nchw_fchw">

    %conv4, %matmul = transform.structured.convert_conv2d_to_img2col %conv3
      : (!transform.op<"linalg.conv_2d_nchw_fchw">) -> (!transform.any_op, !transform.any_op)

    transform.debug.emit_remark_at %conv4, "img2col" : !transform.any_op

    transform.apply_patterns to %conv4 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.linalg.tiling_canonicalization
    } : !transform.any_op

    transform.yield
  }

}
