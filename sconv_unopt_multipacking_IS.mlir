!input_tensor_t = tensor<1x128x66x66xf32>
!weight_tensor_t = tensor<256x128x3x3xf32>
!output_tensor_t = tensor<1x256x64x64xf32>

#map = affine_map<(d0) -> (d0 floordiv 64)>
#map1 = affine_map<(d0) -> (d0 mod 64)>

#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0) -> (d0 floordiv 9)>
#map4 = affine_map<(d0) -> ((d0 mod 9) floordiv 3)>
#map5 = affine_map<(d0, d1) -> (d0 + d1 mod 3)>

#map6 = affine_map<(d0, d1) -> (d0 * 8 + d1)>
#map7 = affine_map<(d0) -> (d0 mod 3)>
#map8 = affine_map<(d0) -> (d0 floordiv 8)>

#map9 = affine_map<(d0, d1, d2, d3) -> (d3, d1)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

module {
  func.func @conv_2d_nchw_fchw(%arg0: tensor<1x128x66x66xf32>, %arg1: tensor<256x128x3x3xf32>, %arg2: tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32> {
    %collapsed = tensor.collapse_shape %arg2 [[0], [1], [2, 3]] : tensor<1x256x64x64xf32> into tensor<1x256x4096xf32>
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %c0_2 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c4096 = arith.constant 4096 : index
    %c256 = arith.constant 256 : index
    %c1_3 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %0 = scf.for %arg3 = %c0 to %c1 step %c1_3 iter_args(%arg4 = %collapsed) -> (tensor<1x256x4096xf32>) {
      %1 = scf.for %arg5 = %c0_0 to %c128 step %c16 iter_args(%arg6 = %arg4) -> (tensor<1x256x4096xf32>) {
        %2 = scf.for %arg7 = %c0_2 to %c256 step %c64 iter_args(%arg8 = %arg6) -> (tensor<1x256x4096xf32>) {
          %3 = scf.for %arg9 = %c0_1 to %c4096 step %c32 iter_args(%arg10 = %arg8) -> (tensor<1x256x4096xf32>) {
            %4 = affine.apply #map(%arg9)
            %5 = affine.apply #map1(%arg9)
            %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg5, %4, %5] [1, 16, 3, 34] [1, 1, 1, 1] : tensor<1x128x66x66xf32> to tensor<1x16x3x34xf32>
            %extracted_slice_4 = tensor.extract_slice %arg1[%arg7, %arg5, 0, 0] [64, 16, 3, 3] [1, 1, 1, 1] : tensor<256x128x3x3xf32> to tensor<64x16x3x3xf32>
            %extracted_slice_5 = tensor.extract_slice %arg10[%arg3, %arg7, %arg9] [1, 64, 32] [1, 1, 1] : tensor<1x256x4096xf32> to tensor<1x64x32xf32>
            %c0_6 = arith.constant 0 : index
            %c0_7 = arith.constant 0 : index
            %c64_8 = arith.constant 64 : index
            %c32_9 = arith.constant 32 : index
            %c8 = arith.constant 8 : index
            %c16_10 = arith.constant 16 : index

            %18 = tensor.empty() : tensor<16x144x8xf32>
            %19 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%18 : tensor<16x144x8xf32>) {
            ^bb0(%out: f32):
              %20 = linalg.index 0 : index
              %21 = linalg.index 1 : index
              %22 = linalg.index 2 : index
              %23 = affine.apply #map6(%20, %22)
              %24 = affine.apply #map3(%21)
              %25 = affine.apply #map4(%21)
              %26 = affine.apply #map7(%21)
              %extracted = tensor.extract %extracted_slice_4[%23, %24, %25, %26] : tensor<64x16x3x3xf32>
              linalg.yield %extracted : f32
            } -> tensor<16x144x8xf32>

            %6 = scf.for %arg11 = %c0_7 to %c32_9 step %c16_10 iter_args(%arg12 = %extracted_slice_5) -> (tensor<1x64x32xf32>) {

              %8 = affine.apply #map(%arg11)
              %9 = affine.apply #map1(%arg11)
              %extracted_slice_11 = tensor.extract_slice %extracted_slice[0, 0, %8, %9] [1, 16, 3, 18] [1, 1, 1, 1] : tensor<1x16x3x34xf32> to tensor<1x16x3x18xf32>


              %10 = tensor.empty() : tensor<1x144x16xf32>
              %11 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%10 : tensor<1x144x16xf32>) {
              ^bb0(%out: f32):
                %12 = linalg.index 0 : index
                %13 = linalg.index 1 : index
                %14 = linalg.index 2 : index
                %15 = affine.apply #map3(%13)
                %16 = affine.apply #map4(%13)
                %17 = affine.apply #map5(%14, %13)
                %extracted = tensor.extract %extracted_slice_11[%12, %15, %16, %17] : tensor<1x16x3x18xf32>
                linalg.yield %extracted : f32
              } -> tensor<1x144x16xf32>

              %7 = scf.for %arg13 = %c0_6 to %c64_8 step %c8 iter_args(%arg14 = %arg12) -> (tensor<1x64x32xf32>) {
                %27 = affine.apply #map8(%arg13)
                %extracted_slice_12 = tensor.extract_slice %19[%27, 0, 0] [1, 144, 8] [1, 1, 1] : tensor<16x144x8xf32> to tensor<1x144x8xf32>
                %extracted_slice_13 = tensor.extract_slice %arg14[0, %arg13, %arg11] [1, 8, 16] [1, 1, 1] : tensor<1x64x32xf32> to tensor<1x8x16xf32>

                %collapsed_2 = tensor.collapse_shape %extracted_slice_12 [[0, 1], [2]] : tensor<1x144x8xf32> into tensor<144x8xf32>


                %28 = linalg.generic {indexing_maps = [#map9, #map10, #map11], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_2, %11 : tensor<144x8xf32>, tensor<1x144x16xf32>) outs(%extracted_slice_13 : tensor<1x8x16xf32>) {
                ^bb0(%in: f32, %in_5: f32, %out: f32):
                  %29 = arith.mulf %in, %in_5 : f32
                  %30 = arith.addf %29, %out : f32
                  linalg.yield %30 : f32
                } -> tensor<1x8x16xf32>

                %inserted_slice_14 = tensor.insert_slice %28 into %arg14[0, %arg13, %arg11] [1, 8, 16] [1, 1, 1] : tensor<1x8x16xf32> into tensor<1x64x32xf32>
                scf.yield %inserted_slice_14 : tensor<1x64x32xf32>
              }
              scf.yield %7 : tensor<1x64x32xf32>
            }
            %inserted_slice = tensor.insert_slice %6 into %arg10[%arg3, %arg7, %arg9] [1, 64, 32] [1, 1, 1] : tensor<1x64x32xf32> into tensor<1x256x4096xf32>
            scf.yield %inserted_slice : tensor<1x256x4096xf32>
          }
          scf.yield %3 : tensor<1x256x4096xf32>
        }
        scf.yield %2 : tensor<1x256x4096xf32>
      }
      scf.yield %1 : tensor<1x256x4096xf32>
    }
    %expanded = tensor.expand_shape %0 [[0], [1], [2, 3]] output_shape [1, 256, 64, 64] : tensor<1x256x4096xf32> into tensor<1x256x64x64xf32>
    return %expanded : tensor<1x256x64x64xf32>
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
