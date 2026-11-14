package nn

import (
	"goForge/autoGrad"
	"goForge/tensor"
	"testing"
)

func TestMLP_Forward(t *testing.T) {
	m := &MLP{
		l1: &Linear{
			W: autoGrad.NewVar(tensor.NewTensor([]float32{1, 1, 1, 1}, 2, 2)),
			B: autoGrad.NewVar(tensor.NewTensor([]float32{0, -1}, 2)),
		},
		l2: &Linear{
			W: autoGrad.NewVar(tensor.NewTensor([]float32{1, -2}, 2, 1)),
			B: autoGrad.NewVar(tensor.NewTensor([]float32{0}, 1)),
		},
	}
	x := autoGrad.NewVar(tensor.NewTensor([]float32{0, 0, 0, 1, 1, 0, 1, 1}, 4, 2))

	out := m.Forward(x)
	want := []float32{0, 1, 1, 0}

	for i := range want {
		if out.Data.Data[i] != want[i] {
			t.Errorf("want %v, got %v", want, out.Data.Data)
		}
	}
}
