// The original conv_2d_nchw_fchw

// !input_tensor_t = tensor<1x128x66x66xf32>
// !weight_tensor_t = tensor<256x128x3x3xf32>
// !output_tensor_t = tensor<1x256x64x64xf32>

// RUN: mlir-opt conv.mlir -linalg-generalize-named-ops -o gconv.mlir
// func.func @conv_2d_nchw_fchw(%in: !input_tensor_t, %wei: !weight_tensor_t,
//                              %out: !output_tensor_t) -> !output_tensor_t {
//   %res = linalg.conv_2d_nchw_fchw
//     {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
//      ins(%in, %wei: !input_tensor_t, !weight_tensor_t)
//     outs(%out: !output_tensor_t) -> !output_tensor_t
//   return %res : !output_tensor_t
// }

// Below, the original conv.mlir, after applying linalg pass generalize-named-ops

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

// The linalg.generic above is manually modified to address SConv transformations
// First, the output tensor is linearized in wo & ho dimensions (64x64) to who (4096) one-dim
// affine_maps d2 (64) = wo and d3 (64) = ho is modified to d3 (4096) to represent who
// Then, the collapse_shape and expand_shape is inserted at beginning & end of function to preserve the interface

#map = affine_map<(d0, d1, d3, d4, d5, d6) -> (d0, d4, d3 floordiv 64 + d5, d3 mod 64 + d6)>
#map1 = affine_map<(d0, d1, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d3, d4, d5, d6) -> (d0, d1, d3)>

!input_tensor_t = tensor<1x128x66x66xf32>
!weight_tensor_t = tensor<256x128x3x3xf32>
!output_tensor_t = tensor<1x256x64x64xf32>
!linearized_tensor_t = tensor<1x256x4096xf32>

module {
  func.func @conv_2d_nchw_fchw(%arg0: !input_tensor_t, %arg1: !weight_tensor_t,
                               %arg2: !output_tensor_t) -> !output_tensor_t {
    %collapsed = tensor.collapse_shape %arg2 [[0], [1], [2, 3]] : !output_tensor_t into !linearized_tensor_t
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : !input_tensor_t, !weight_tensor_t) outs(%collapsed : !linearized_tensor_t) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> !linearized_tensor_t
    %expanded = tensor.expand_shape %0 [[0], [1], [2, 3]] output_shape [1, 256, 64, 64] : !linearized_tensor_t into !output_tensor_t
    return %expanded : !output_tensor_t
  }

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Create tensors
    // TODO: Eliminate repeated values 
    %inp = tensor.generate {
    ^bb0(%n : index, %c : index, %h : index, %w : index):
      %m1 = arith.addi %w, %h : index
      %m2 = arith.addi %m1, %c : index
      %m3 = arith.addi %m2, %n : index
      %i64 = arith.index_cast %m3 : index to i64
      %f32 = arith.uitofp %i64 : i64 to f32
      tensor.yield %f32 : f32
    } : !input_tensor_t

    %wei = tensor.generate {
    ^bb0(%f : index, %c : index, %h : index, %w : index):
      %m1 = arith.addi %w, %h : index
      %m2 = arith.addi %m1, %c : index
      %m3 = arith.addi %m2, %f : index
      %i64 = arith.index_cast %m3 : index to i64
      %f32 = arith.uitofp %i64 : i64 to f32
      tensor.yield %f32 : f32
    } : !weight_tensor_t

    %out = tensor.generate {
    ^bb0(%n : index, %f : index, %h : index, %w : index):
      %m1 = arith.addi %w, %h : index
      %m2 = arith.addi %m1, %f : index
      %m3 = arith.addi %m2, %n : index
      %i64 = arith.index_cast %m3 : index to i64
      %f32 = arith.uitofp %i64 : i64 to f32
      tensor.yield %f32 : f32
    } : !output_tensor_t

    // Number of iterations
    %num_reps = arith.constant 1 : index

    // Run convolution and time it
    %t_start = call @rtclock() : () -> f64
    %final_res = scf.for %arg0 = %c0 to %num_reps step %c1
      iter_args(%out_loop = %out) -> (!output_tensor_t) {
      %res = func.call @conv_2d_nchw_fchw(%inp, %wei, %out_loop)
        : (!input_tensor_t, !weight_tensor_t, !output_tensor_t)
        -> !output_tensor_t
      scf.yield %res : !output_tensor_t
    }
    %t_end = call @rtclock() : () -> f64
    %t = arith.subf %t_end, %t_start : f64

    // Print the result
    %un_res = tensor.cast %final_res : !output_tensor_t to tensor<*xf32>
    func.call @printMemrefF32(%un_res) : (tensor<*xf32>) -> ()

    // num_flops_per_iter = 2 * OH * OW * F * C * KH * KW
    %num_flops_per_iter = arith.constant 2415919104 : index

    // num_flops_total = num_flops_per_iter * num_reps
    %num_flops_total = arith.muli %num_flops_per_iter, %num_reps: index

    // Print the number of flops per second
    %num_flops_total_i = arith.index_cast %num_flops_total : index to i64
    %num_flops_total_f = arith.uitofp %num_flops_total_i : i64 to f64
    %flops_per_s = arith.divf %num_flops_total_f, %t : f64
    call @printFlops(%flops_per_s) : (f64) -> ()

    return
  }

  func.func private @printMemrefF32(tensor<*xf32>) attributes { llvm.emit_c_interface }
  func.func private @printFlops(f64)
  func.func private @rtclock() -> f64
}
