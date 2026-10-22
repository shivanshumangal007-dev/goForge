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

//func multiplyShapeCheck(shapeA, shapeB []int) bool {
//	if len(shapeA) != len(shapeB) {
//		return false
//	}
//
//	l := len(shapeA)
//	if shapeA[l-1] != shapeB[l-2]{
//		return false
//	}
//
//	return true
//}
