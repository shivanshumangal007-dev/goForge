package tensor

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

func sameShape(shapeA, shapeB []int) bool {
	if len(shapeA) != len(shapeB) {
		return false
	}

	for i, v := range shapeA {
		if v != shapeB[i] {
			return false
		}
	}

	return true
}
func (t *Tensor) IsContiguous() bool {
	want := computeStrides(t.shape)
	for i := range want {
		if t.stride[i] != want[i] {
			return false
		}
	}
	return true
}

// Contiguous returns t itself if already contiguous, else a packed copy.
func Contiguous(t *Tensor) *Tensor {
	if t.IsContiguous() {
		return t
	}
	if len(t.shape) != 2 {
		panic("tensor.Contiguous: only 2D for now")
	}
	rows, cols := t.shape[0], t.shape[1]
	out := make([]float32, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			out[i*cols+j] = t.At(i, j)
		}
	}
	return NewTensor(out, rows, cols)
}
