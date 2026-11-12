package optim

import (
	"goForge/autoGrad"
	"goForge/tensor"
	"math"
	"testing"
)

func TestStep(t *testing.T) {
	param := &autoGrad.Var{
		Data: tensor.NewTensor([]float32{1, 2}, 2),
		Grad: tensor.NewTensor([]float32{10, 20}, 2),
	}
	Nsgd := &Sgd{
		params: []*autoGrad.Var{param},
		lr:     float32(0.1),
	}
	Nsgd.Step()
	want := []float32{0, 0}
	wantG := []float32{10, 20}
	for i := range want {
		if math.Abs(float64(Nsgd.params[0].Data.Data[i]-want[i])) > 1e-6 {
			t.Fatalf("Data = %v, want %v", param.Data.Data, want)
		}
		if Nsgd.params[0].Grad.Data[i] != wantG[i] {
			t.Fatalf("Grad = %v, want %v", Nsgd.params[0].Grad.Data, wantG[i])
		}
	}
}

func TestSgd_ZeroGrad(t *testing.T) {
	param := &autoGrad.Var{
		Data: tensor.NewTensor([]float32{1, 2}, 2),
		Grad: tensor.NewTensor([]float32{10, 20}, 2),
	}
	Nsgd := &Sgd{
		params: []*autoGrad.Var{param},
		lr:     float32(0.1),
	}
	Nsgd.ZeroGrad()
	want := []float32{0, 0}
	for i := range want {
		if math.Abs(float64(Nsgd.params[0].Grad.Data[i]-want[i])) > 1e-6 {
			t.Fatalf("Data = %v, want %v", param.Data.Data, want)
		}
	}
}
func TestSgd_negativeSlope(t *testing.T) {
	param := &autoGrad.Var{
		Data: tensor.NewTensor([]float32{0}, 1),
		Grad: tensor.NewTensor([]float32{-5}, 1),
	}
	Nsgd := &Sgd{
		params: []*autoGrad.Var{param},
		lr:     float32(0.1),
	}

	Nsgd.Step()
	want := []float32{0.5}
	for i := range want {
		if math.Abs(float64(Nsgd.params[0].Data.Data[i]-want[i])) > 1e-6 {
			t.Fatalf("Data = %v, want %v", Nsgd.params[0].Data.Data, want)
		}
	}
}

func TestSgd_twoParams(t *testing.T) {
	param1 := &autoGrad.Var{
		Data: tensor.NewTensor([]float32{1, 2}, 2),
		Grad: tensor.NewTensor([]float32{10, 20}, 2),
	}
	param2 := &autoGrad.Var{
		Data: tensor.NewTensor([]float32{3, 4}, 2),
		Grad: tensor.NewTensor([]float32{30, 40}, 2),
	}
	Nsgd := &Sgd{
		params: []*autoGrad.Var{param1, param2},
		lr:     float32(0.1),
	}
	Nsgd.Step()
	want := []float32{0, 0}
	for _, p := range Nsgd.params {
		for i, d := range p.Data.Data {
			if float64(d-want[i]) > 1e-6 {
				t.Fatalf("Data = %v, want %v", p.Data.Data, want)
			}
		}
	}
}
