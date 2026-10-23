package autoGrad

import "testing"

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
