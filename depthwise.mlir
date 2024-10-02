// RUN: mlir-opt depthwise.mlir --pass-pipeline="builtin.module(transform-interpreter{debug-bind-trailing-args=linalg.depthwise_conv_2d_nhwc_hwcm},canonicalize,cse,symbol-dce)" -o depthwise_gen.mlir

// Original depthwise_conv_2d_nhwc_hwcm to be converted into generic.
func.func @depthwise_conv_2d_nhwc_hwcm(%input: tensor<2x4x5x2xf32>, %filter: tensor<2x2x2x3xf32>, %output: tensor<2x3x4x2x3xf32>) {
  %depw = linalg.depthwise_conv_2d_nhwc_hwcm {
      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
  }
  ins(%input, %filter : tensor<2x4x5x2xf32>, tensor<2x2x2x3xf32>)
  outs(%output : tensor<2x3x4x2x3xf32>) -> tensor<2x3x4x2x3xf32>
  return %depw : tensor<2x3x4x2x3xf32>
}

module attributes {transform.with_named_sequence} {

transform.named_sequence @__transform_main(
  %arg0: !transform.any_op,
  %conv: !transform.op<"linalg.depthwise_conv_2d_nhwc_hwcm">) {

    transform.debug.emit_remark_at %conv, "depthwise conv" : !transform.op<"linalg.depthwise_conv_2d_nhwc_hwcm">
  
    %res = transform.structured.generalize %conv
    : (!transform.op<"linalg.depthwise_conv_2d_nhwc_hwcm">)
    -> !transform.any_op 
  
    transform.debug.emit_remark_at %res, "result" : !transform.any_op
  
    transform.yield
  }
}
