package nn

import (
	"goForge/autoGrad"
	"math/rand"
)

type MLP struct {
	l1 *Linear
	l2 *Linear
}

func NewMLP(in, hidden, out int, rng *rand.Rand) *MLP {
	return &MLP{
		l1: NewLinear(in, hidden, rng),
		l2: NewLinear(hidden, out, rng),
	}
}

func (m *MLP) Forward(x *autoGrad.Var) *autoGrad.Var {
	//step1
	h1 := m.l1.Forward(x)
	//step2
	h2 := autoGrad.ReLuVar(h1)
	//step3
	out := m.l2.Forward(h2)
	return out
}

func (m *MLP) Params() []*autoGrad.Var {
	a := []*autoGrad.Var{m.l1.W, m.l1.B, m.l2.W, m.l2.B}
	return a
}
