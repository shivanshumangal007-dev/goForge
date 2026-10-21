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
	for i, v := range b.data{
		outData[i] += v;
	}

	return NewTensor(outData, a.shape...)
}
