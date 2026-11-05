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
func (v *Var) Backward() {
	visited := make(map[*Var]bool)
	order := make([]*Var, 0)
	var visit func(*Var)
	visit = func(n *Var) {
		if visited[n] {
			return
		}
		visited[n] = true
		for _, p := range n.prev {
			visit(p)
		}
		order = append(order, n)
	}

	visit(v)

	for i := range v.Grad.Data {
		v.Grad.Data[i] = 1
	}
	for i := len(order) - 1; i >= 0; i-- {
		if order[i].backward != nil {
			order[i].backward()
		}
	}
}

func Sum(v *Var) *Var {
	sum := float32(0)
	for _, x := range v.Data.Data {
		sum += x
	}
	out := &Var{
		Data: tensor.NewTensor([]float32{sum}, 1),
		Grad: tensor.Zeros(1),
		prev: []*Var{v},
	}
	out.backward = func() {
		for i := range v.Grad.Data {
			v.Grad.Data[i] += out.Grad.Data[0]
		}
	}

	return out
}

func AddVar(a, b *Var) *Var {
	out := &Var{
		Data: tensor.Add(a.Data, b.Data),
		prev: []*Var{a, b},
	}
	out.Grad = tensor.Zeros(out.Data.Shape()...)

	out.backward = func() {
		tensor.AddInPlace(a.Grad, out.Grad)
		tensor.AddInPlace(b.Grad, out.Grad)
	}
	return out
}
func MulVar(a, b *Var) *Var {
	t := tensor.Zeros(a.Data.Shape()...)
	for i := range a.Data.Data {
		t.Data[i] = a.Data.Data[i] * b.Data.Data[i]
	}

	out := &Var{
		Data: t,
		Grad: tensor.Zeros(a.Grad.Shape()...),
		prev: []*Var{a, b},
	}

	out.backward = func() {
		for i := range out.Grad.Data {
			a.Grad.Data[i] += out.Grad.Data[i] * b.Data.Data[i]
			b.Grad.Data[i] += out.Grad.Data[i] * a.Data.Data[i]
		}
	}

	return out
}

func ReLuVar(a *Var) *Var {
	out := &Var{
		Data: a.Data.Relu(),
		Grad: tensor.Zeros(a.Grad.Shape()...),
		prev: []*Var{a},
	}

	out.backward = func() {
		for i := range out.Grad.Data {
			if a.Data.Data[i] > 0 {
				a.Grad.Data[i] += out.Grad.Data[i]
			}
		}
	}

	return out
}
func MatmulVar(a, b *Var) *Var {
	out := &Var{
		Data: tensor.Multiply(a.Data, b.Data),
		prev: []*Var{a, b},
	}

	out.Grad = tensor.Zeros(out.Data.Shape()...)
	out.backward = func() {
		dA := tensor.Multiply(out.Grad, tensor.Transpose(b.Data))
		dB := tensor.Multiply(tensor.Transpose(a.Data), out.Grad)

		tensor.AddInPlace(a.Grad, dA)
		tensor.AddInPlace(b.Grad, dB)
	}

	return out
}
