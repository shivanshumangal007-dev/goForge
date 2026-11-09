package autoGrad

import (
	"goForge/tensor"
	"testing"
)

func TestNewVar(t *testing.T) {
	tens := tensor.NewTensor([]float32{1, 2, 4, 5, 6, 7}, 2, 3)
	d := NewVar(tens)
	gs := d.Grad.Shape()
	if gs[0] != 2 || gs[1] != 3 {
		t.Fatalf("grad shape = %v, want [2 3]", gs)
	}
	for _, v := range d.Grad.Data {
		if v != 0 {
			t.Fatal("gradient value is not zero")
		}
	}
}
func TestSumForward(t *testing.T) {
	a := NewVar(tensor.NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3))
	out := Sum(a)

	if out.Data.Data[0] != 21 {
		t.Fatalf("got %v, want 21", out.Data.Data[0])
	}
}

func TestSumBackward(t *testing.T) {
	a := NewVar(tensor.NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3))
	out := Sum(a)
	out.Backward()

	for i, g := range a.Grad.Data {
		if g != 1 {
			t.Fatalf("a.Grad[%d] = %v, want 1", i, g)
		}
	}
}
func TestAddVarForward(t *testing.T) {
	a := NewVar(tensor.NewTensor([]float32{1, 2, 3, 4}, 2, 2))
	b := NewVar(tensor.NewTensor([]float32{10, 20, 30, 40}, 2, 2))

	out := AddVar(a, b)

	want := []float32{11, 22, 33, 44}
	for i := range want {
		if out.Data.Data[i] != want[i] {
			t.Fatalf("got %v, want %v", out.Data.Data, want)
		}
	}
}

func TestAddVarBackward(t *testing.T) {
	a := NewVar(tensor.NewTensor([]float32{1, 2, 3, 4}, 2, 2))
	b := NewVar(tensor.NewTensor([]float32{10, 20, 30, 40}, 2, 2))

	Sum(AddVar(a, b)).Backward()

	for i := range a.Grad.Data {
		if a.Grad.Data[i] != 1 || b.Grad.Data[i] != 1 {
			t.Fatalf("a.Grad=%v b.Grad=%v, want all 1", a.Grad.Data, b.Grad.Data)
		}
	}
}

func TestAddVarReuse(t *testing.T) {
	a := NewVar(tensor.NewTensor([]float32{1, 2, 3, 4}, 2, 2))

	Sum(AddVar(a, a)).Backward()

	for i, g := range a.Grad.Data {
		if g != 2 {
			t.Fatalf("a.Grad[%d] = %v, want 2", i, g)
		}
	}
}

func TestAddVarShapeMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	a := NewVar(tensor.Zeros(2, 2))
	b := NewVar(tensor.Zeros(2, 3))
	AddVar(a, b)
}

func TestMulVarForward(t *testing.T) {
	a := NewVar(tensor.NewTensor([]float32{1, 2, 3, 4}, 2, 2))
	b := NewVar(tensor.NewTensor([]float32{5, 6, 7, 8}, 2, 2))

	out := MulVar(a, b)
	want := []float32{5, 12, 21, 32}

	for i := range out.Data.Data {
		if out.Data.Data[i] != want[i] {
			t.Fatalf("got %v, want %v", out.Grad.Data, want)
		}
	}
}

func TestMulVarBackward(t *testing.T) {
	a := NewVar(tensor.NewTensor([]float32{1, 2, 3, 4}, 2, 2))
	b := NewVar(tensor.NewTensor([]float32{5, 6, 7, 8}, 2, 2))

	out := MulVar(a, b)
	out.Backward()
	for i := range out.Grad.Data {
		if out.Grad.Data[i] != 1 {
			t.Fatalf("a.Grad[%d] = %v, want 1", i, out.Grad.Data)
		}
	}

	for i := range a.Grad.Data {
		if a.Grad.Data[i] != b.Data.Data[i] {
			t.Fatalf("a.Grad[%d] = %f, want %f", i, a.Grad.Data[i], b.Data.Data[i])
		}
		if b.Grad.Data[i] != a.Data.Data[i] {
			t.Fatalf("b.Grad[%d] = %f, want %f", i, b.Grad.Data[i], a.Data.Data[i])
		}
	}
}
func TestMulVarReuse(t *testing.T) {
	a := NewVar(tensor.NewTensor([]float32{1, 2, 3, 4}, 2, 2))

	Sum(MulVar(a, a)).Backward()

	want := []float32{2, 4, 6, 8}
	for i := range want {
		if a.Grad.Data[i] != want[i] {
			t.Fatalf("a.Grad = %v, want %v", a.Grad.Data, want)
		}
	}
}
func TestReLUVarForwardBackward(t *testing.T) {
	a := NewVar(tensor.NewTensor([]float32{-1, 2, 0, 3}, 2, 2))
	out := ReLuVar(a)

	wantOut := []float32{0, 2, 0, 3}
	for i := range wantOut {
		if out.Data.Data[i] != wantOut[i] {
			t.Fatalf("forward: got %v, want %v", out.Data.Data, wantOut)
		}
	}

	// the input must be untouched
	if a.Data.Data[0] != -1 {
		t.Fatalf("ReLU modified its input: %v", a.Data.Data)
	}

	Sum(out).Backward()

	wantGrad := []float32{0, 1, 0, 1}
	for i := range wantGrad {
		if a.Grad.Data[i] != wantGrad[i] {
			t.Fatalf("backward: got %v, want %v", a.Grad.Data, wantGrad)
		}
	}
}

func TestReLUVarAllNegative(t *testing.T) {
	a := NewVar(tensor.NewTensor([]float32{-2, -1, -3, -4}, 2, 2))

	Sum(ReLuVar(a)).Backward()

	for i, g := range a.Grad.Data {
		if g != 0 {
			t.Fatalf("a.Grad[%d] = %v, want 0", i, g)
		}
	}
}

func TestMultiplyVarBackward(t *testing.T) {
	a := NewVar(tensor.NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3))
	b := NewVar(tensor.NewTensor([]float32{7, 8, 9, 10, 11, 12}, 3, 2))
	L := Sum(MatmulVar(a, b))
	L.Backward()
	wantL := []float32{415}
	wantA := []float32{15, 19, 23, 15, 19, 23}
	wantB := []float32{5, 5, 7, 7, 9, 9}

	for i := range wantL {
		if L.Data.Data[i] != wantL[i] {
			t.Fatalf("multiply: got %v, want %v", L.Data.Data, wantL)
		}
	}

	for i := range wantA {
		if a.Grad.Data[i] != wantA[i] {
			t.Fatalf("a.Grad[%d] = %v, want %v", i, a.Grad.Data, wantA)
		}
	}
	for i := range wantB {
		if b.Grad.Data[i] != wantB[i] {
			t.Fatalf("a.Grad[%d] = %v, want %v", i, b.Grad.Data, wantB)
		}
	}
}
func TestMultiplyVarForward(t *testing.T) {
	a := NewVar(tensor.NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3))
	b := NewVar(tensor.NewTensor([]float32{7, 8, 9, 10, 11, 12}, 3, 2))
	L := MatmulVar(a, b)
	L.Backward()

	wantL := []float32{58, 64, 139, 154}

	for i := range wantL {
		if L.Data.Data[i] != wantL[i] {
			t.Fatalf("multiply: got %v, want %v", L.Data.Data, wantL)
		}
	}
}
func TestMeanVar(t *testing.T) {
	a := NewVar(tensor.NewTensor([]float32{1, 2, 3, 4}, 2, 2))
	out := MeanVar(a)

	out.Backward()
	if out.Data.Data[0] != 2.5 {
		t.Fatalf("mean: got %v, want 2.5", out.Data.Data[0])
	}
	for i := range a.Grad.Data {
		if a.Grad.Data[i] != 0.25 {
			t.Fatalf("mean: got %v, want 0.25", out.Grad.Data[i])
		}
	}
}
func TestAddBias(t *testing.T) {
	x := NewVar(tensor.NewTensor([]float32{1, 2, 3, 4}, 2, 2))
	b := NewVar(tensor.NewTensor([]float32{10, 20}, 2))

	out := AddBias(x, b)

	want := []float32{11, 22, 13, 24}
	for i := range want {
		if out.Data.Data[i] != want[i] {
			t.Fatalf("forward: got %v, want %v", out.Data.Data, want)
		}
	}

	Sum(out).Backward()

	for i, g := range x.Grad.Data {
		if g != 1 {
			t.Fatalf("x.Grad[%d] = %v, want 1", i, g)
		}
	}
	if b.Grad.Data[0] != 2 || b.Grad.Data[1] != 2 {
		t.Fatalf("bias.Grad = %v, want [2 2]", b.Grad.Data)
	}
}
