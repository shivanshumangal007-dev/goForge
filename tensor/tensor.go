package tensor

type Tensor struct {
	Data   []float32
	shape  []int // [rows, coloumn]
	stride []int
}

func NewTensor(Data []float32, shape ...int) *Tensor {
	totalEle := num_el(shape)

	if len(Data) != totalEle {
		panic("tensor.New: Data length does not match shape")
	}

	return &Tensor{Data: Data, shape: shape, stride: computeStrides(shape)}
}

func (t *Tensor) At(positions ...int) float32 {
	offset := 0
	for i, v := range positions {
		offset += t.stride[i] * v
	}

	return t.Data[offset]
}

func Add(a, b *Tensor) *Tensor {
	if !sameShape(a.shape, b.shape) {
		panic("tensor: shape of a and b is not the same")
	}
	outData := make([]float32, len(a.Data))
	copy(outData, a.Data)
	for i, v := range b.Data {
		outData[i] += v
	}

	return NewTensor(outData, a.shape...)
}
func Sub(a, b *Tensor) *Tensor {
	if !sameShape(a.shape, b.shape) {
		panic("tensor: shape of a and b is not the same")
	}
	outData := make([]float32, len(a.Data))
	copy(outData, a.Data)
	for i, v := range b.Data {
		outData[i] -= v
	}

	return NewTensor(outData, a.shape...)
}
func Multiply(a *Tensor, b *Tensor) *Tensor {
	a = Contiguous(a)
	b = Contiguous(b)
	if len(a.shape) != 2 || len(b.shape) != 2 {
		panic("tensor.Matmul: both tensors must be 2D")
	}
	m, k := a.shape[0], a.shape[1]
	k2, n := b.shape[0], b.shape[1]
	if k != k2 {
		panic("tensor.Matmul: inner dimensions do not match")
	}

	out := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for p := 0; p < k; p++ {
				sum += a.Data[i*k+p] * b.Data[p*n+j]
			}
			out[i*n+j] = sum
		}
	}
	return NewTensor(out, m, n)
}
func Transpose(t *Tensor) *Tensor {
	if len(t.shape) != 2 {
		panic("tensor.Transpose: shape is not 2")
	}
	return &Tensor{
		Data:   t.Data,
		shape:  []int{t.shape[1], t.shape[0]},
		stride: []int{t.stride[1], t.stride[0]},
	}
}
func ReShape(t *Tensor, shape ...int) *Tensor {
	totalEle := num_el(shape)
	if len(t.Data) != totalEle {
		panic("tensor.ReShape: shape lengths do not match")
	}

	t = Contiguous(t)

	return NewTensor(t.Data, shape...)
}

func (t *Tensor) Shape() []int {
	return t.shape
}

func Zeros(shape ...int) *Tensor {
	l := num_el(shape)
	zeroArr := make([]float32, l)
	return NewTensor(zeroArr, shape...)
}
func AddInPlace(a, b *Tensor) {
	shape1 := a.shape
	shape2 := b.shape
	if len(shape1) != len(shape2) {
		panic("tensor.AddInPlace: shape lengths do not match")
	}
	for i := range shape1 {
		if shape1[i] != shape2[i] {
			panic("tensor.AddInPlace: shape lengths do not match")
		}
	}

	for i, _ := range a.Data {
		a.Data[i] += b.Data[i]
	}
}

func (t *Tensor) Relu() *Tensor {
	newData := make([]float32, len(t.Data))
	for i, d := range t.Data {
		if d > 0 {
			newData[i] = d
			continue
		}
		newData[i] = 0
	}

	return NewTensor(newData, t.shape...)
}
