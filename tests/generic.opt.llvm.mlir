module {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @conv_2d_nchw_fchw(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: !llvm.ptr, %arg23: !llvm.ptr, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64) -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> {
    %0 = llvm.mlir.constant(3 : i64) : i64
    %1 = llvm.mlir.constant(3 : index) : i64
    %2 = llvm.mlir.constant(-1 : index) : i64
    %3 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %4 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %5 = llvm.mlir.constant(4 : i64) : i64
    %6 = llvm.mlir.zero : !llvm.ptr
    %7 = llvm.mlir.constant(1048576 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(0 : index) : i64
    %10 = llvm.mlir.constant(128 : index) : i64
    %11 = llvm.mlir.constant(4096 : index) : i64
    %12 = llvm.mlir.constant(256 : index) : i64
    %13 = llvm.mlir.constant(16 : index) : i64
    %14 = llvm.mlir.constant(32 : index) : i64
    %15 = llvm.mlir.constant(64 : index) : i64
    %16 = llvm.mlir.constant(8 : index) : i64
    %17 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %18 = llvm.insertvalue %arg22, %17[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %19 = llvm.insertvalue %arg23, %18[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %20 = llvm.insertvalue %arg24, %19[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %21 = llvm.insertvalue %arg25, %20[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %22 = llvm.insertvalue %arg29, %21[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %23 = llvm.insertvalue %arg26, %22[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %24 = llvm.insertvalue %arg30, %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %25 = llvm.insertvalue %arg27, %24[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %26 = llvm.insertvalue %arg31, %25[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %27 = llvm.insertvalue %arg28, %26[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %28 = llvm.insertvalue %arg32, %27[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %29 = llvm.getelementptr %6[1048576] : (!llvm.ptr) -> !llvm.ptr, f32
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.add %30, %15 : i64
    %32 = llvm.call @malloc(%31) : (i64) -> !llvm.ptr
    %33 = llvm.ptrtoint %32 : !llvm.ptr to i64
    %34 = llvm.sub %15, %8 : i64
    %35 = llvm.add %33, %34 : i64
    %36 = llvm.urem %35, %15  : i64
    %37 = llvm.sub %35, %36 : i64
    %38 = llvm.inttoptr %37 : i64 to !llvm.ptr
    %39 = llvm.insertvalue %32, %17[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %40 = llvm.insertvalue %38, %39[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %41 = llvm.insertvalue %9, %40[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %42 = llvm.insertvalue %8, %41[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %43 = llvm.insertvalue %12, %42[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %44 = llvm.insertvalue %15, %43[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %45 = llvm.insertvalue %15, %44[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %46 = llvm.insertvalue %7, %45[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %47 = llvm.insertvalue %11, %46[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %48 = llvm.insertvalue %15, %47[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %49 = llvm.insertvalue %8, %48[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %50 = llvm.intr.stacksave : !llvm.ptr
    %51 = llvm.alloca %8 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %28, %51 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %52 = llvm.insertvalue %5, %4[0] : !llvm.struct<(i64, ptr)> 
    %53 = llvm.insertvalue %51, %52[1] : !llvm.struct<(i64, ptr)> 
    %54 = llvm.alloca %8 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %49, %54 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %55 = llvm.insertvalue %54, %52[1] : !llvm.struct<(i64, ptr)> 
    %56 = llvm.alloca %8 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %53, %56 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %57 = llvm.alloca %8 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %55, %57 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %58 = llvm.getelementptr %6[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %59 = llvm.ptrtoint %58 : !llvm.ptr to i64
    llvm.call @memrefCopy(%59, %56, %57) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %50 : !llvm.ptr
    llvm.br ^bb1(%9 : i64)
  ^bb1(%60: i64):  // 2 preds: ^bb0, ^bb32
    %61 = llvm.icmp "slt" %60, %10 : i64
    llvm.cond_br %61, ^bb2, ^bb33
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%9 : i64)
  ^bb3(%62: i64):  // 2 preds: ^bb2, ^bb31
    %63 = llvm.icmp "slt" %62, %11 : i64
    llvm.cond_br %63, ^bb4, ^bb32
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%9 : i64)
  ^bb5(%64: i64):  // 2 preds: ^bb4, ^bb30
    %65 = llvm.icmp "slt" %64, %12 : i64
    llvm.cond_br %65, ^bb6, ^bb31
  ^bb6:  // pred: ^bb5
    %66 = llvm.mul %64, %11 : i64
    %67 = llvm.add %66, %62 : i64
    %68 = llvm.insertvalue %32, %3[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %69 = llvm.insertvalue %38, %68[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %70 = llvm.insertvalue %67, %69[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %71 = llvm.insertvalue %8, %70[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %72 = llvm.insertvalue %7, %71[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %73 = llvm.insertvalue %15, %72[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %74 = llvm.insertvalue %11, %73[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %75 = llvm.insertvalue %14, %74[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %76 = llvm.insertvalue %8, %75[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb7(%9 : i64)
  ^bb7(%77: i64):  // 2 preds: ^bb6, ^bb29
    %78 = llvm.icmp "slt" %77, %15 : i64
    llvm.cond_br %78, ^bb8, ^bb30
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%9 : i64)
  ^bb9(%79: i64):  // 2 preds: ^bb8, ^bb28
    %80 = llvm.icmp "slt" %79, %14 : i64
    llvm.cond_br %80, ^bb10, ^bb29
  ^bb10:  // pred: ^bb9
    %81 = llvm.icmp "slt" %62, %9 : i64
    %82 = llvm.sub %2, %62 : i64
    %83 = llvm.select %81, %82, %62 : i1, i64
    %84 = llvm.sdiv %83, %15  : i64
    %85 = llvm.sub %2, %84 : i64
    %86 = llvm.select %81, %85, %84 : i1, i64
    %87 = llvm.mul %86, %arg9 : i64
    %88 = llvm.mul %60, %arg8 : i64
    %89 = llvm.add %arg2, %88 : i64
    %90 = llvm.add %87, %89 : i64
    %91 = llvm.srem %62, %15  : i64
    %92 = llvm.icmp "slt" %91, %9 : i64
    %93 = llvm.add %91, %15 : i64
    %94 = llvm.select %92, %93, %91 : i1, i64
    %95 = llvm.mul %94, %arg10 : i64
    %96 = llvm.add %90, %95 : i64
    %97 = llvm.icmp "slt" %79, %9 : i64
    %98 = llvm.sub %2, %79 : i64
    %99 = llvm.select %97, %98, %79 : i1, i64
    %100 = llvm.sdiv %99, %15  : i64
    %101 = llvm.sub %2, %100 : i64
    %102 = llvm.select %97, %101, %100 : i1, i64
    %103 = llvm.mul %102, %arg9 : i64
    %104 = llvm.add %103, %96 : i64
    %105 = llvm.srem %79, %15  : i64
    %106 = llvm.icmp "slt" %105, %9 : i64
    %107 = llvm.add %105, %15 : i64
    %108 = llvm.select %106, %107, %105 : i1, i64
    %109 = llvm.mul %108, %arg10 : i64
    %110 = llvm.add %104, %109 : i64
    %111 = llvm.mul %64, %arg18 : i64
    %112 = llvm.add %arg13, %111 : i64
    %113 = llvm.mul %60, %arg19 : i64
    %114 = llvm.add %112, %113 : i64
    %115 = llvm.mul %77, %arg18 : i64
    %116 = llvm.add %114, %115 : i64
    %117 = llvm.mul %77, %11 : i64
    %118 = llvm.add %67, %117 : i64
    %119 = llvm.add %118, %79 : i64
    %120 = llvm.insertvalue %119, %69[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %121 = llvm.insertvalue %8, %120[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %122 = llvm.insertvalue %7, %121[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %123 = llvm.insertvalue %16, %122[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %124 = llvm.insertvalue %11, %123[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %125 = llvm.insertvalue %13, %124[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %126 = llvm.insertvalue %8, %125[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb11(%9 : i64)
  ^bb11(%127: i64):  // 2 preds: ^bb10, ^bb27
    %128 = llvm.icmp "slt" %127, %8 : i64
    llvm.cond_br %128, ^bb12, ^bb28
  ^bb12:  // pred: ^bb11
    llvm.br ^bb13(%9 : i64)
  ^bb13(%129: i64):  // 2 preds: ^bb12, ^bb26
    %130 = llvm.icmp "slt" %129, %16 : i64
    llvm.cond_br %130, ^bb14, ^bb27
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15(%9 : i64)
  ^bb15(%131: i64):  // 2 preds: ^bb14, ^bb25
    %132 = llvm.icmp "slt" %131, %13 : i64
    llvm.cond_br %132, ^bb16, ^bb26
  ^bb16:  // pred: ^bb15
    llvm.br ^bb17(%9 : i64)
  ^bb17(%133: i64):  // 2 preds: ^bb16, ^bb24
    %134 = llvm.icmp "slt" %133, %13 : i64
    llvm.cond_br %134, ^bb18, ^bb25
  ^bb18:  // pred: ^bb17
    llvm.br ^bb19(%9 : i64)
  ^bb19(%135: i64):  // 2 preds: ^bb18, ^bb23
    %136 = llvm.icmp "slt" %135, %1 : i64
    llvm.cond_br %136, ^bb20, ^bb24
  ^bb20:  // pred: ^bb19
    llvm.br ^bb21(%9 : i64)
  ^bb21(%137: i64):  // 2 preds: ^bb20, ^bb22
    %138 = llvm.icmp "slt" %137, %1 : i64
    llvm.cond_br %138, ^bb22, ^bb23
  ^bb22:  // pred: ^bb21
    %139 = llvm.add %131, %137 : i64
    %140 = llvm.getelementptr %arg1[%110] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %141 = llvm.mul %127, %arg7 : i64
    %142 = llvm.mul %133, %arg8 : i64
    %143 = llvm.add %141, %142 : i64
    %144 = llvm.mul %135, %arg9 : i64
    %145 = llvm.add %143, %144 : i64
    %146 = llvm.mul %139, %arg10 : i64
    %147 = llvm.add %145, %146 : i64
    %148 = llvm.getelementptr %140[%147] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %149 = llvm.load %148 : !llvm.ptr -> f32
    %150 = llvm.getelementptr %arg12[%116] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %151 = llvm.mul %129, %arg18 : i64
    %152 = llvm.mul %133, %arg19 : i64
    %153 = llvm.add %151, %152 : i64
    %154 = llvm.mul %135, %arg20 : i64
    %155 = llvm.add %153, %154 : i64
    %156 = llvm.mul %137, %arg21 : i64
    %157 = llvm.add %155, %156 : i64
    %158 = llvm.getelementptr %150[%157] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %159 = llvm.load %158 : !llvm.ptr -> f32
    %160 = llvm.getelementptr %38[%119] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %161 = llvm.mul %127, %7 : i64
    %162 = llvm.mul %129, %11 : i64
    %163 = llvm.add %161, %162 : i64
    %164 = llvm.add %163, %131 : i64
    %165 = llvm.getelementptr %160[%164] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %166 = llvm.load %165 : !llvm.ptr -> f32
    %167 = llvm.fmul %149, %159  : f32
    %168 = llvm.fadd %166, %167  : f32
    llvm.store %168, %165 : f32, !llvm.ptr
    %169 = llvm.add %137, %8 : i64
    llvm.br ^bb21(%169 : i64)
  ^bb23:  // pred: ^bb21
    %170 = llvm.add %135, %8 : i64
    llvm.br ^bb19(%170 : i64)
  ^bb24:  // pred: ^bb19
    %171 = llvm.add %133, %8 : i64
    llvm.br ^bb17(%171 : i64)
  ^bb25:  // pred: ^bb17
    %172 = llvm.add %131, %8 : i64
    llvm.br ^bb15(%172 : i64)
  ^bb26:  // pred: ^bb15
    %173 = llvm.add %129, %8 : i64
    llvm.br ^bb13(%173 : i64)
  ^bb27:  // pred: ^bb13
    %174 = llvm.add %127, %8 : i64
    llvm.br ^bb11(%174 : i64)
  ^bb28:  // pred: ^bb11
    %175 = llvm.intr.stacksave : !llvm.ptr
    %176 = llvm.alloca %8 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %126, %176 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %177 = llvm.insertvalue %0, %4[0] : !llvm.struct<(i64, ptr)> 
    %178 = llvm.insertvalue %176, %177[1] : !llvm.struct<(i64, ptr)> 
    %179 = llvm.alloca %8 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %126, %179 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %180 = llvm.insertvalue %179, %177[1] : !llvm.struct<(i64, ptr)> 
    %181 = llvm.alloca %8 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %178, %181 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %182 = llvm.alloca %8 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %180, %182 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @memrefCopy(%59, %181, %182) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %175 : !llvm.ptr
    %183 = llvm.add %79, %13 : i64
    llvm.br ^bb9(%183 : i64)
  ^bb29:  // pred: ^bb9
    %184 = llvm.add %77, %16 : i64
    llvm.br ^bb7(%184 : i64)
  ^bb30:  // pred: ^bb7
    %185 = llvm.intr.stacksave : !llvm.ptr
    %186 = llvm.alloca %8 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %76, %186 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %187 = llvm.insertvalue %0, %4[0] : !llvm.struct<(i64, ptr)> 
    %188 = llvm.insertvalue %186, %187[1] : !llvm.struct<(i64, ptr)> 
    %189 = llvm.alloca %8 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %76, %189 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %190 = llvm.insertvalue %189, %187[1] : !llvm.struct<(i64, ptr)> 
    %191 = llvm.alloca %8 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %188, %191 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %192 = llvm.alloca %8 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %190, %192 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @memrefCopy(%59, %191, %192) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %185 : !llvm.ptr
    %193 = llvm.add %64, %15 : i64
    llvm.br ^bb5(%193 : i64)
  ^bb31:  // pred: ^bb5
    %194 = llvm.add %62, %14 : i64
    llvm.br ^bb3(%194 : i64)
  ^bb32:  // pred: ^bb3
    %195 = llvm.add %60, %13 : i64
    llvm.br ^bb1(%195 : i64)
  ^bb33:  // pred: ^bb1
    llvm.return %49 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant(4 : index) : i64
    %1 = llvm.mlir.constant(1048576 : index) : i64
    %2 = llvm.mlir.constant(4096 : index) : i64
    %3 = llvm.mlir.constant(1152 : index) : i64
    %4 = llvm.mlir.constant(9 : index) : i64
    %5 = llvm.mlir.constant(3 : index) : i64
    %6 = llvm.mlir.constant(256 : index) : i64
    %7 = llvm.mlir.constant(0 : index) : i64
    %8 = llvm.mlir.constant(64 : index) : i64
    %9 = llvm.mlir.constant(0x41E2000000000000 : f64) : f64
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.mlir.constant(128 : index) : i64
    %12 = llvm.mlir.constant(66 : index) : i64
    %13 = llvm.mlir.constant(4356 : index) : i64
    %14 = llvm.mlir.constant(557568 : index) : i64
    %15 = llvm.mlir.zero : !llvm.ptr
    %16 = llvm.getelementptr %15[557568] : (!llvm.ptr) -> !llvm.ptr, f32
    %17 = llvm.ptrtoint %16 : !llvm.ptr to i64
    %18 = llvm.add %17, %8 : i64
    %19 = llvm.call @malloc(%18) : (i64) -> !llvm.ptr
    %20 = llvm.ptrtoint %19 : !llvm.ptr to i64
    %21 = llvm.sub %8, %10 : i64
    %22 = llvm.add %20, %21 : i64
    %23 = llvm.urem %22, %8  : i64
    %24 = llvm.sub %22, %23 : i64
    %25 = llvm.inttoptr %24 : i64 to !llvm.ptr
    llvm.br ^bb1(%7 : i64)
  ^bb1(%26: i64):  // 2 preds: ^bb0, ^bb11
    %27 = llvm.icmp "slt" %26, %10 : i64
    llvm.cond_br %27, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%7 : i64)
  ^bb3(%28: i64):  // 2 preds: ^bb2, ^bb10
    %29 = llvm.icmp "slt" %28, %11 : i64
    llvm.cond_br %29, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%7 : i64)
  ^bb5(%30: i64):  // 2 preds: ^bb4, ^bb9
    %31 = llvm.icmp "slt" %30, %12 : i64
    llvm.cond_br %31, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%7 : i64)
  ^bb7(%32: i64):  // 2 preds: ^bb6, ^bb8
    %33 = llvm.icmp "slt" %32, %12 : i64
    llvm.cond_br %33, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %34 = llvm.add %32, %30 : i64
    %35 = llvm.add %34, %28 : i64
    %36 = llvm.add %35, %26 : i64
    %37 = llvm.uitofp %36 : i64 to f32
    %38 = llvm.mul %26, %14 : i64
    %39 = llvm.mul %28, %13 : i64
    %40 = llvm.add %38, %39 : i64
    %41 = llvm.mul %30, %12 : i64
    %42 = llvm.add %40, %41 : i64
    %43 = llvm.add %42, %32 : i64
    %44 = llvm.getelementptr %25[%43] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %37, %44 : f32, !llvm.ptr
    %45 = llvm.add %32, %10 : i64
    llvm.br ^bb7(%45 : i64)
  ^bb9:  // pred: ^bb7
    %46 = llvm.add %30, %10 : i64
    llvm.br ^bb5(%46 : i64)
  ^bb10:  // pred: ^bb5
    %47 = llvm.add %28, %10 : i64
    llvm.br ^bb3(%47 : i64)
  ^bb11:  // pred: ^bb3
    %48 = llvm.add %26, %10 : i64
    llvm.br ^bb1(%48 : i64)
  ^bb12:  // pred: ^bb1
    %49 = llvm.getelementptr %15[294912] : (!llvm.ptr) -> !llvm.ptr, f32
    %50 = llvm.ptrtoint %49 : !llvm.ptr to i64
    %51 = llvm.add %50, %8 : i64
    %52 = llvm.call @malloc(%51) : (i64) -> !llvm.ptr
    %53 = llvm.ptrtoint %52 : !llvm.ptr to i64
    %54 = llvm.add %53, %21 : i64
    %55 = llvm.urem %54, %8  : i64
    %56 = llvm.sub %54, %55 : i64
    %57 = llvm.inttoptr %56 : i64 to !llvm.ptr
    llvm.br ^bb13(%7 : i64)
  ^bb13(%58: i64):  // 2 preds: ^bb12, ^bb23
    %59 = llvm.icmp "slt" %58, %6 : i64
    llvm.cond_br %59, ^bb14, ^bb24
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15(%7 : i64)
  ^bb15(%60: i64):  // 2 preds: ^bb14, ^bb22
    %61 = llvm.icmp "slt" %60, %11 : i64
    llvm.cond_br %61, ^bb16, ^bb23
  ^bb16:  // pred: ^bb15
    llvm.br ^bb17(%7 : i64)
  ^bb17(%62: i64):  // 2 preds: ^bb16, ^bb21
    %63 = llvm.icmp "slt" %62, %5 : i64
    llvm.cond_br %63, ^bb18, ^bb22
  ^bb18:  // pred: ^bb17
    llvm.br ^bb19(%7 : i64)
  ^bb19(%64: i64):  // 2 preds: ^bb18, ^bb20
    %65 = llvm.icmp "slt" %64, %5 : i64
    llvm.cond_br %65, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %66 = llvm.add %64, %62 : i64
    %67 = llvm.add %66, %60 : i64
    %68 = llvm.add %67, %58 : i64
    %69 = llvm.uitofp %68 : i64 to f32
    %70 = llvm.mul %58, %3 : i64
    %71 = llvm.mul %60, %4 : i64
    %72 = llvm.add %70, %71 : i64
    %73 = llvm.mul %62, %5 : i64
    %74 = llvm.add %72, %73 : i64
    %75 = llvm.add %74, %64 : i64
    %76 = llvm.getelementptr %57[%75] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %69, %76 : f32, !llvm.ptr
    %77 = llvm.add %64, %10 : i64
    llvm.br ^bb19(%77 : i64)
  ^bb21:  // pred: ^bb19
    %78 = llvm.add %62, %10 : i64
    llvm.br ^bb17(%78 : i64)
  ^bb22:  // pred: ^bb17
    %79 = llvm.add %60, %10 : i64
    llvm.br ^bb15(%79 : i64)
  ^bb23:  // pred: ^bb15
    %80 = llvm.add %58, %10 : i64
    llvm.br ^bb13(%80 : i64)
  ^bb24:  // pred: ^bb13
    %81 = llvm.getelementptr %15[1048576] : (!llvm.ptr) -> !llvm.ptr, f32
    %82 = llvm.ptrtoint %81 : !llvm.ptr to i64
    %83 = llvm.add %82, %8 : i64
    %84 = llvm.call @malloc(%83) : (i64) -> !llvm.ptr
    %85 = llvm.ptrtoint %84 : !llvm.ptr to i64
    %86 = llvm.add %85, %21 : i64
    %87 = llvm.urem %86, %8  : i64
    %88 = llvm.sub %86, %87 : i64
    %89 = llvm.inttoptr %88 : i64 to !llvm.ptr
    llvm.br ^bb25(%7 : i64)
  ^bb25(%90: i64):  // 2 preds: ^bb24, ^bb35
    %91 = llvm.icmp "slt" %90, %10 : i64
    llvm.cond_br %91, ^bb26, ^bb36
  ^bb26:  // pred: ^bb25
    llvm.br ^bb27(%7 : i64)
  ^bb27(%92: i64):  // 2 preds: ^bb26, ^bb34
    %93 = llvm.icmp "slt" %92, %6 : i64
    llvm.cond_br %93, ^bb28, ^bb35
  ^bb28:  // pred: ^bb27
    llvm.br ^bb29(%7 : i64)
  ^bb29(%94: i64):  // 2 preds: ^bb28, ^bb33
    %95 = llvm.icmp "slt" %94, %8 : i64
    llvm.cond_br %95, ^bb30, ^bb34
  ^bb30:  // pred: ^bb29
    llvm.br ^bb31(%7 : i64)
  ^bb31(%96: i64):  // 2 preds: ^bb30, ^bb32
    %97 = llvm.icmp "slt" %96, %8 : i64
    llvm.cond_br %97, ^bb32, ^bb33
  ^bb32:  // pred: ^bb31
    %98 = llvm.add %96, %94 : i64
    %99 = llvm.add %98, %92 : i64
    %100 = llvm.add %99, %90 : i64
    %101 = llvm.uitofp %100 : i64 to f32
    %102 = llvm.mul %90, %1 : i64
    %103 = llvm.mul %92, %2 : i64
    %104 = llvm.add %102, %103 : i64
    %105 = llvm.mul %94, %8 : i64
    %106 = llvm.add %104, %105 : i64
    %107 = llvm.add %106, %96 : i64
    %108 = llvm.getelementptr %89[%107] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %101, %108 : f32, !llvm.ptr
    %109 = llvm.add %96, %10 : i64
    llvm.br ^bb31(%109 : i64)
  ^bb33:  // pred: ^bb31
    %110 = llvm.add %94, %10 : i64
    llvm.br ^bb29(%110 : i64)
  ^bb34:  // pred: ^bb29
    %111 = llvm.add %92, %10 : i64
    llvm.br ^bb27(%111 : i64)
  ^bb35:  // pred: ^bb27
    %112 = llvm.add %90, %10 : i64
    llvm.br ^bb25(%112 : i64)
  ^bb36:  // pred: ^bb25
    %113 = llvm.call @rtclock() : () -> f64
    %114 = llvm.call @conv_2d_nchw_fchw(%19, %25, %7, %10, %11, %12, %12, %14, %13, %12, %10, %52, %57, %7, %6, %11, %5, %5, %3, %4, %5, %10, %84, %89, %7, %10, %6, %8, %8, %1, %2, %8, %10) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %115 = llvm.call @rtclock() : () -> f64
    %116 = llvm.fsub %115, %113  : f64
    %117 = llvm.alloca %10 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %114, %117 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    llvm.call @printMemrefF32(%0, %117) : (i64, !llvm.ptr) -> ()
    %118 = llvm.fdiv %9, %116  : f64
    llvm.call @printFlops(%118) : (f64) -> ()
    llvm.return
  }
  llvm.func private @printMemrefF32(%arg0: i64, %arg1: !llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %2 = llvm.insertvalue %arg0, %1[0] : !llvm.struct<(i64, ptr)> 
    %3 = llvm.insertvalue %arg1, %2[1] : !llvm.struct<(i64, ptr)> 
    %4 = llvm.alloca %0 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %3, %4 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @_mlir_ciface_printMemrefF32(%4) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_printMemrefF32(!llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @printFlops(f64) attributes {sym_visibility = "private"}
  llvm.func @rtclock() -> f64 attributes {sym_visibility = "private"}
}

