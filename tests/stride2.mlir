// RUN: mlir-opt "$1".mlir -linalg-generalize-named-ops -o "$1"_gen.mlir

!input_tensor_t = tensor<1x256x56x56xf32>
!weight_tensor_t = tensor<128x256x1x1xf32>
!output_tensor_t = tensor<1x128x28x28xf32>

// Convolução nomeada NCHW com strides [2, 2] a ser generalizada
func.func @conv_2d_nchw_fchw(%in: !input_tensor_t, %wei: !weight_tensor_t,
                             %out: !output_tensor_t) -> !output_tensor_t {
  %res = linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
     ins(%in, %wei: !input_tensor_t, !weight_tensor_t)
    outs(%out: !output_tensor_t) -> !output_tensor_t
  return %res : !output_tensor_t
}
