; RUN: not llvm-as -opaque-pointers -disable-output %s 2>&1 | FileCheck %s

define i32 @f() jumptable {
  ret i32 0
}

; CHECK: Attribute 'jumptable' requires 'unnamed_addr'
; CHECK: ptr @f
