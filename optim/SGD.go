package optim

import "goForge/autoGrad"

type Sgd struct {
	params []*autoGrad.Var
	lr     float32
}

func NewSgd(pr []*autoGrad.Var, lr float32) *Sgd {
	return &Sgd{
		params: pr,
		lr:     lr,
	}
}

func (sgd *Sgd) Step() {
	for _, p := range sgd.params {
		for i := range p.Data.Data {
			p.Data.Data[i] -= sgd.lr * p.Grad.Data[i]
		}
	}
}

func (sgd *Sgd) ZeroGrad() {
	for _, p := range sgd.params {
		for i := range p.Data.Data {
			p.Grad.Data[i] = 0
		}
	}
}
