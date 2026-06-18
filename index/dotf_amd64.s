//go:build amd64 && !noasm

#include "textflag.h"

// func dotProductAVX(a, b []float32) float32
//
// Inner product of two float32 slices of equal length, where the length is a
// multiple of 8 (the Go dispatcher in dotf_amd64.go guarantees this and handles
// the tail). Four independent 256-bit accumulators keep the floating-point
// pipeline full; the bulk loop processes 32 floats per iteration.
TEXT ·dotProductAVX(SB), NOSPLIT, $0-52
	MOVQ a_base+0(FP), SI
	MOVQ b_base+24(FP), DI
	MOVQ a_len+8(FP), CX      // n, a multiple of 8
	XORQ AX, AX               // i = 0
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3

	MOVQ CX, R8
	ANDQ $-32, R8             // R8 = n rounded down to a multiple of 32
	CMPQ AX, R8
	JGE  tail

loop32:
	VMOVUPS 0(SI)(AX*4), Y4
	VMOVUPS 32(SI)(AX*4), Y5
	VMOVUPS 64(SI)(AX*4), Y6
	VMOVUPS 96(SI)(AX*4), Y7
	VMULPS  0(DI)(AX*4), Y4, Y4
	VMULPS  32(DI)(AX*4), Y5, Y5
	VMULPS  64(DI)(AX*4), Y6, Y6
	VMULPS  96(DI)(AX*4), Y7, Y7
	VADDPS  Y4, Y0, Y0
	VADDPS  Y5, Y1, Y1
	VADDPS  Y6, Y2, Y2
	VADDPS  Y7, Y3, Y3
	ADDQ    $32, AX
	CMPQ    AX, R8
	JL      loop32

tail:
	CMPQ    AX, CX
	JGE     reduce

loop8:
	VMOVUPS 0(SI)(AX*4), Y4
	VMULPS  0(DI)(AX*4), Y4, Y4
	VADDPS  Y4, Y0, Y0
	ADDQ    $8, AX
	CMPQ    AX, CX
	JL      loop8

reduce:
	VADDPS  Y1, Y0, Y0
	VADDPS  Y3, Y2, Y2
	VADDPS  Y2, Y0, Y0           // Y0 = 8 partial sums
	VEXTRACTF128 $1, Y0, X1
	VADDPS  X1, X0, X0           // X0 = 4 partial sums
	VHADDPS X0, X0, X0           // 2
	VHADDPS X0, X0, X0           // 1
	MOVSS   X0, ret+48(FP)
	VZEROUPPER
	RET
