module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(
    %arg0: !transform.any_op) {
    %conv = transform.structured.match ops{["linalg.generic"]} in %arg0
      : (!transform.any_op) -> !transform.op<"linalg.generic">

    // transform.debug.emit_remark_at %conv, "Input conv" : !transform.op<"linalg.generic">

    %conv2, %loops2:4 = transform.structured.tile_using_for %conv
      // N, F, OHW, C, KH, KW
      tile_sizes [1, 64, 32, 16, 0, 0]   // 16 C , 32 OHW, 64 F
      interchange = [0, 3, 2, 1] // F, OHW, C
      : (!transform.op<"linalg.generic">)
      -> (!transform.op<"linalg.generic">, !transform.any_op, 
          !transform.any_op, !transform.any_op, !transform.any_op)

    // transform.debug.emit_remark_at %conv2, "conv2" : !transform.op<"linalg.generic">

 %conv3, %loops3:2 = transform.structured.tile_using_for %conv2
   // N, F, OHW, C, KH, KW
   tile_sizes [0, 8, 16, 0, 0, 0]
   interchange = [1, 0]
   : (!transform.op<"linalg.generic">)
   -> (!transform.op<"linalg.generic">, !transform.any_op,
       !transform.any_op)

//  transform.debug.emit_remark_at %conv3, "conv3" : !transform.op<"linalg.generic">

    transform.apply_patterns to %conv3 {
    transform.apply_patterns.canonicalization
     transform.apply_patterns.linalg.tiling_canonicalization
    } : !transform.op<"linalg.generic">

    transform.yield
  }

}
