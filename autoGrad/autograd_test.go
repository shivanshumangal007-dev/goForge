package autoGrad

import (
	"math"
	"testing"
)

func TestMulBackward(t *testing.T) {
	a := NewVal(2)
	b := NewVal(3)
	out := Mul(a, b)

	out.Backward()

	if a.Grad != 3 || b.Grad != 2 {
		t.Fatalf("got a.Grad=%v b.Grad=%v, want 3 and 2", a.Grad, b.Grad)
	}
}
func TestMulReuse(t *testing.T) {
	a := NewVal(4)
	out := Mul(a, a)

	out.Backward()

	if a.Grad != 8 {
		t.Fatalf("a.Grad = %v, want 6", a.Grad)
	}
}

func TestBackward(t *testing.T) {
	a := NewVal(4)
	b := NewVal(3)
	m := Mul(a, b)
	c := NewVal(2)
	d := Add(m, c)

	d.Backward()
	if a.Grad != 3 || b.Grad != 4 || c.Grad != 1 {
		t.Fatalf("got %v %v %v, want 3 4 1", a.Grad, b.Grad, c.Grad)
	}
}
func TestSub(t *testing.T) {
	a := NewVal(4)
	b := NewVal(3)
	m := Mul(a, b)
	c := NewVal(2)
	d := Sub(m, c)

	d.Backward()

	if a.Grad != 3 || b.Grad != 4 || c.Grad != -1 {
		t.Fatalf("got %v %v %v, want 3 4 -1", a.Grad, b.Grad, c.Grad)
	}

}
func TestPow(t *testing.T) {
	a := NewVal(4)
	d := Pow(a, 2)

	d.Backward()

	if a.Grad != 8 || d.Grad != 1 {
		t.Fatalf("expected a.grad=1 and d.grad=8 got %f and %f", a.Grad, d.Grad)
	}
}

func TestReLU(t *testing.T) {
	a := NewVal(3)
	b := NewVal(-2)
	ReLU(a).Backward()
	ReLU(b).Backward()
	if a.Grad != 1 || b.Grad != 0 {
		t.Fatalf("got %v %v, want 1 0", a.Grad, b.Grad)
	}
}

func TestTanh(t *testing.T) {
	a := NewVal(0)
	b := NewVal(1)

	Tanh(a).Backward()
	Tanh(b).Backward()

	if math.Abs(a.Grad-1) > 1e-9 {
		t.Fatalf("tanh at 0: got grad %v, want 1", a.Grad)
	}
	if math.Abs(b.Grad-0.4200) > 1e-3 {
		t.Fatalf("tanh at 1: got grad %v, want about 0.42", b.Grad)
	}
}
