package autoGrad

import "goForge/tensor"

type Var struct {
	Data     *tensor.Tensor
	Grad     *tensor.Tensor
	prev     []*Var
	backward func()
}

func NewVar(t *tensor.Tensor) *Var {
	return &Var{
		Data: t,
		Grad: tensor.Zeros(t.Shape()...),
	}
}
