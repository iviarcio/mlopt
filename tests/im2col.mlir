// RUN: mlir-opt tests/im2col.mlir --pass-pipeline="builtin.module(transform-interpreter{debug-bind-trailing-args=linalg.conv_2d_nchw_fchw},canonicalize,cse,symbol-dce)" -o ressults/im2col_res.mlir

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

    %conv2, %matmul = transform.structured.convert_conv2d_to_img2col %conv
      : (!transform.op<"linalg.conv_2d_nchw_fchw">) -> (!transform.any_op, !transform.any_op)

    transform.debug.emit_remark_at %conv2, "img2col" : !transform.any_op

    transform.apply_patterns to %conv2 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.linalg.tiling_canonicalization
    } : !transform.any_op

    transform.yield
  }

}
