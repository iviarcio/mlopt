#map = affine_map<(d0) -> (d0 floordiv 64)>
#map1 = affine_map<(d0) -> (d0 mod 64)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d2 floordiv 64 + d4, d2 mod 64 + d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d4, d5)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
module {
  module {
    func.func @conv_2d_nchw_fchw(%arg0: tensor<1x128x66x66xf32>, %arg1: tensor<256x128x3x3xf32>, %arg2: tensor<1x256x4096xf32>) -> tensor<1x256x4096xf32> {
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c4096 = arith.constant 4096 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %0 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %arg2) -> (tensor<1x256x4096xf32>) {
        %1 = scf.for %arg5 = %c0 to %c4096 step %c32 iter_args(%arg6 = %arg4) -> (tensor<1x256x4096xf32>) {
          %2 = scf.for %arg7 = %c0 to %c256 step %c64 iter_args(%arg8 = %arg6) -> (tensor<1x256x4096xf32>) {
            %3 = affine.apply #map(%arg5)
            %4 = affine.apply #map1(%arg5)
            %extracted_slice = tensor.extract_slice %arg0[0, %arg3, %3, %4] [1, 16, 3, 34] [1, 1, 1, 1] : tensor<1x128x66x66xf32> to tensor<1x16x3x34xf32>
            %extracted_slice_0 = tensor.extract_slice %arg1[%arg7, %arg3, 0, 0] [64, 16, 3, 3] [1, 1, 1, 1] : tensor<256x128x3x3xf32> to tensor<64x16x3x3xf32>
            %extracted_slice_1 = tensor.extract_slice %arg8[0, %arg7, %arg5] [1, 64, 32] [1, 1, 1] : tensor<1x256x4096xf32> to tensor<1x64x32xf32>
            %5 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice, %extracted_slice_0 : tensor<1x16x3x34xf32>, tensor<64x16x3x3xf32>) outs(%extracted_slice_1 : tensor<1x64x32xf32>) {
            ^bb0(%in: f32, %in_2: f32, %out: f32):
              %6 = arith.mulf %in, %in_2 : f32
              %7 = arith.addf %out, %6 : f32
              linalg.yield %7 : f32
            } -> tensor<1x64x32xf32>
            %inserted_slice = tensor.insert_slice %5 into %arg8[0, %arg7, %arg5] [1, 64, 32] [1, 1, 1] : tensor<1x64x32xf32> into tensor<1x256x4096xf32>
            scf.yield %inserted_slice : tensor<1x256x4096xf32>
          }
          scf.yield %2 : tensor<1x256x4096xf32>
        }
        scf.yield %1 : tensor<1x256x4096xf32>
      }
      return %0 : tensor<1x256x4096xf32>
    }
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op, %arg1: !transform.op<"linalg.generic">) {
      %tiled_linalg_op, %loops:4 = transform.structured.tile_using_for %arg1 tile_sizes [1, 64, 32, 16, 0, 0] interchange = [0, 3, 2, 1] : (!transform.op<"linalg.generic">) -> (!transform.op<"linalg.generic">, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield 
    }
  }
}

