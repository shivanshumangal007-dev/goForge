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
