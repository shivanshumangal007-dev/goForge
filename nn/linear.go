package nn

import (
	"goForge/autoGrad"
	"goForge/tensor"
	"math/rand"
)

type Linear struct {
	W *autoGrad.Var
	B *autoGrad.Var
}

func NewLinear(in int, out int, rng *rand.Rand) *Linear {
	var Wdata []float32
	for range in * out {
		Wdata = append(Wdata, float32(rng.NormFloat64()*0.1))
	}
	return &Linear{
		W: autoGrad.NewVar(tensor.NewTensor(Wdata, in, out)),
		B: autoGrad.NewVar(tensor.Zeros(out)),
	}
}
func (l *Linear) Forward(a *autoGrad.Var) *autoGrad.Var {
	sa := a.Data.Shape()
	if len(sa) != 2 || sa[1] != l.W.Data.Shape()[0] {
		panic("nn.Linear: input must be [n, in]")
	}
	out1 := autoGrad.MatmulVar(a, l.W)
	finalOut := autoGrad.AddBias(out1, l.B)

	return finalOut
}
func (l *Linear) Params() []*autoGrad.Var {
	return []*autoGrad.Var{l.W, l.B}
}
