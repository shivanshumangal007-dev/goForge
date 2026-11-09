package nn

import (
	"goForge/autoGrad"
	"goForge/tensor"
	"math/rand"
	"testing"
)

func TestLinear_Forward(t *testing.T) {
	l := &Linear{
		W: autoGrad.NewVar(tensor.NewTensor([]float32{1, 2}, 2, 1)),
		B: autoGrad.NewVar(tensor.NewTensor([]float32{0.5}, 1)),
	}
	a := autoGrad.NewVar(tensor.NewTensor([]float32{1, 2, 3, 4}, 2, 2))
	out := l.Forward(a)

	if out.Data.At(0, 0) != 5.5 || out.Data.At(0, 1) != 11.5 {
		t.Fatalf("Forward = %v, want 5.5", out.Data.Data[0])
	}
}
func TestNewLinear(t *testing.T) {
	rng1 := rand.New(rand.NewSource(0))
	rng2 := rand.New(rand.NewSource(0))

	l1 := NewLinear(2, 1, rng1)
	l2 := NewLinear(2, 1, rng2)

	for i, v := range l1.W.Data.Data {
		if v != l2.W.Data.Data[i] {
			t.Fatalf("W of l1 and l2 do not match")
		}
	}
}
