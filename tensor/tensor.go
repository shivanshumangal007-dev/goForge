package tensor

type Tensor struct {
	data   []float32
	shape  []int // [rows, coloumn]
	stride []int
}

func NewTensor(data []float32, shape ...int) *Tensor {
	totalEle := num_el(shape)

	if len(data) != totalEle {
		panic("tensor.New: data length does not match shape")
	}

	return &Tensor{data: data, shape: shape, stride: computeStrides(shape)}
}

func (t *Tensor) At(positions ...int) float32 {
	offset := 0
	for i, v := range positions {
		offset += t.stride[i] * v
	}

	return t.data[offset]
}

func Add(a, b *Tensor) *Tensor {
	if !sameShape(a.shape, b.shape) {
		panic("tensor: shape of a and b is not the same")
	}
	outData := make([]float32, len(a.data))
	copy(outData, a.data)
	for i, v := range b.data {
		outData[i] += v
	}

	return NewTensor(outData, a.shape...)
}
func Multiply(a, b *Tensor) *Tensor {
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
				sum += a.data[i*k+p] * b.data[p*n+j]
			}
			out[i*n+j] = sum
		}
	}
	return NewTensor(out, m, n)
}
