package tensor

type Tensor struct {
	data   []float32
	shape  []int // [rows, coloumn]
	stride []int
}

func NewTensor(data []float32, shape ...int) *Tensor {
	totalEle := num_el(shape)

	if len(data) < totalEle {
		panic("tensor.New: data length does not match shape")
	}

	return &Tensor{data: data, shape: shape, stride: computeStrides(shape)}
}

// num_el = total number of elements for a shape.
func num_el(shape []int) int {
	n := 1
	for _, s := range shape {
		n *= s
	}
	return n
}

// computeStrides: for shape [2,3] -> strides [3,1].
// Moving one row jumps 3 items, moving one column jumps 1.
func computeStrides(shape []int) []int {
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

func (t *Tensor) At(positions ...int) float32 {
	offset := 0
	for i, v := range positions {
		offset += t.stride[i] * v
	}

	return t.data[offset]
}
