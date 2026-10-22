package autoGrad

type Value struct {
	Data     float64
	Grad     float64
	prev     []*Value
	backward func()
}

func NewVal(Data float64) *Value {
	return &Value{
		Data: Data,
	}
}

func (v *Value) Backward() {
	visited := make(map[*Value]bool)
	order := make([]*Value, 0)
	var visit func(*Value)
	visit = func(n *Value) {
		if visited[n] {
			return
		}
		visited[n] = true
		for _, prev := range n.prev {
			visit(prev)
		}
		order = append(order, n)
	}

	visit(v)

	v.Grad = 1
	for i := len(order) - 1; i >= 0; i-- {

		if order[i].backward != nil {
			order[i].backward()
		}
	}
}
func Add(a, b *Value) *Value {
	out := &Value{
		Data: a.Data + b.Data,
		prev: []*Value{a, b},
	}
	out.backward = func() {
		a.Grad += out.Grad
		b.Grad += out.Grad
	}
	return out
}

func Mul(a, b *Value) *Value {
	out := &Value{
		Data: a.Data * b.Data,
		prev: []*Value{a, b},
	}

	out.backward = func() {
		a.Grad += b.Data * out.Grad
		b.Grad += a.Data * out.Grad
	}
	return out
}
