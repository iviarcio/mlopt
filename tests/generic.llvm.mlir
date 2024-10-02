module {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @conv_2d_nchw_fchw(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: !llvm.ptr, %arg23: !llvm.ptr, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64) -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> {
    %0 = llvm.mlir.constant(-64 : index) : i64
    %1 = llvm.mlir.constant(-1 : index) : i64
    %2 = llvm.mlir.constant(3 : index) : i64
    %3 = llvm.mlir.constant(128 : index) : i64
    %4 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %5 = llvm.mlir.constant(4 : i64) : i64
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.zero : !llvm.ptr
    %8 = llvm.mlir.constant(1048576 : index) : i64
    %9 = llvm.mlir.constant(4096 : index) : i64
    %10 = llvm.mlir.constant(64 : index) : i64
    %11 = llvm.mlir.constant(256 : index) : i64
    %12 = llvm.mlir.constant(1 : index) : i64
    %13 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %14 = llvm.insertvalue %arg22, %13[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %15 = llvm.insertvalue %arg23, %14[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %16 = llvm.insertvalue %arg24, %15[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %17 = llvm.insertvalue %arg25, %16[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %18 = llvm.insertvalue %arg29, %17[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %19 = llvm.insertvalue %arg26, %18[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %20 = llvm.insertvalue %arg30, %19[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %21 = llvm.insertvalue %arg27, %20[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %22 = llvm.insertvalue %arg31, %21[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %23 = llvm.insertvalue %arg28, %22[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %24 = llvm.insertvalue %arg32, %23[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %25 = llvm.getelementptr %7[1048576] : (!llvm.ptr) -> !llvm.ptr, f32
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %27 = llvm.add %26, %10 : i64
    %28 = llvm.call @malloc(%27) : (i64) -> !llvm.ptr
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.sub %10, %12 : i64
    %31 = llvm.add %29, %30 : i64
    %32 = llvm.urem %31, %10  : i64
    %33 = llvm.sub %31, %32 : i64
    %34 = llvm.inttoptr %33 : i64 to !llvm.ptr
    %35 = llvm.insertvalue %28, %13[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %36 = llvm.insertvalue %34, %35[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %37 = llvm.insertvalue %6, %36[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %38 = llvm.insertvalue %12, %37[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %39 = llvm.insertvalue %11, %38[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %40 = llvm.insertvalue %10, %39[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %41 = llvm.insertvalue %10, %40[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %42 = llvm.insertvalue %8, %41[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %43 = llvm.insertvalue %9, %42[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %44 = llvm.insertvalue %10, %43[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %45 = llvm.insertvalue %12, %44[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %46 = llvm.intr.stacksave : !llvm.ptr
    %47 = llvm.alloca %12 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %24, %47 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %48 = llvm.insertvalue %5, %4[0] : !llvm.struct<(i64, ptr)> 
    %49 = llvm.insertvalue %47, %48[1] : !llvm.struct<(i64, ptr)> 
    %50 = llvm.alloca %12 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %45, %50 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %51 = llvm.insertvalue %50, %48[1] : !llvm.struct<(i64, ptr)> 
    %52 = llvm.alloca %12 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %49, %52 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %53 = llvm.alloca %12 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %51, %53 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %54 = llvm.getelementptr %7[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    llvm.call @memrefCopy(%55, %52, %53) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %46 : !llvm.ptr
    llvm.br ^bb1(%6 : i64)
  ^bb1(%56: i64):  // 2 preds: ^bb0, ^bb17
    %57 = llvm.icmp "slt" %56, %12 : i64
    llvm.cond_br %57, ^bb2, ^bb18
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%6 : i64)
  ^bb3(%58: i64):  // 2 preds: ^bb2, ^bb16
    %59 = llvm.icmp "slt" %58, %11 : i64
    llvm.cond_br %59, ^bb4, ^bb17
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%6 : i64)
  ^bb5(%60: i64):  // 2 preds: ^bb4, ^bb15
    %61 = llvm.icmp "slt" %60, %9 : i64
    llvm.cond_br %61, ^bb6, ^bb16
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%6 : i64)
  ^bb7(%62: i64):  // 2 preds: ^bb6, ^bb14
    %63 = llvm.icmp "slt" %62, %3 : i64
    llvm.cond_br %63, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%6 : i64)
  ^bb9(%64: i64):  // 2 preds: ^bb8, ^bb13
    %65 = llvm.icmp "slt" %64, %2 : i64
    llvm.cond_br %65, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%6 : i64)
  ^bb11(%66: i64):  // 2 preds: ^bb10, ^bb12
    %67 = llvm.icmp "slt" %66, %2 : i64
    llvm.cond_br %67, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %68 = llvm.icmp "slt" %60, %6 : i64
    %69 = llvm.sub %1, %60 : i64
    %70 = llvm.select %68, %69, %60 : i1, i64
    %71 = llvm.sdiv %70, %10  : i64
    %72 = llvm.sub %1, %71 : i64
    %73 = llvm.select %68, %72, %71 : i1, i64
    %74 = llvm.add %64, %73 : i64
    %75 = llvm.add %60, %66 : i64
    %76 = llvm.mul %73, %0 : i64
    %77 = llvm.add %75, %76 : i64
    %78 = llvm.getelementptr %arg1[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %79 = llvm.mul %56, %arg7 : i64
    %80 = llvm.mul %62, %arg8 : i64
    %81 = llvm.add %79, %80 : i64
    %82 = llvm.mul %74, %arg9 : i64
    %83 = llvm.add %81, %82 : i64
    %84 = llvm.mul %77, %arg10 : i64
    %85 = llvm.add %83, %84 : i64
    %86 = llvm.getelementptr %78[%85] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %87 = llvm.load %86 : !llvm.ptr -> f32
    %88 = llvm.getelementptr %arg12[%arg13] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %89 = llvm.mul %58, %arg18 : i64
    %90 = llvm.mul %62, %arg19 : i64
    %91 = llvm.add %89, %90 : i64
    %92 = llvm.mul %64, %arg20 : i64
    %93 = llvm.add %91, %92 : i64
    %94 = llvm.mul %66, %arg21 : i64
    %95 = llvm.add %93, %94 : i64
    %96 = llvm.getelementptr %88[%95] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %97 = llvm.load %96 : !llvm.ptr -> f32
    %98 = llvm.mul %56, %8 : i64
    %99 = llvm.mul %58, %9 : i64
    %100 = llvm.add %98, %99 : i64
    %101 = llvm.add %100, %60 : i64
    %102 = llvm.getelementptr %34[%101] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %103 = llvm.load %102 : !llvm.ptr -> f32
    %104 = llvm.fmul %87, %97  : f32
    %105 = llvm.fadd %103, %104  : f32
    llvm.store %105, %102 : f32, !llvm.ptr
    %106 = llvm.add %66, %12 : i64
    llvm.br ^bb11(%106 : i64)
  ^bb13:  // pred: ^bb11
    %107 = llvm.add %64, %12 : i64
    llvm.br ^bb9(%107 : i64)
  ^bb14:  // pred: ^bb9
    %108 = llvm.add %62, %12 : i64
    llvm.br ^bb7(%108 : i64)
  ^bb15:  // pred: ^bb7
    %109 = llvm.add %60, %12 : i64
    llvm.br ^bb5(%109 : i64)
  ^bb16:  // pred: ^bb5
    %110 = llvm.add %58, %12 : i64
    llvm.br ^bb3(%110 : i64)
  ^bb17:  // pred: ^bb3
    %111 = llvm.add %56, %12 : i64
    llvm.br ^bb1(%111 : i64)
  ^bb18:  // pred: ^bb1
    llvm.return %45 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
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

