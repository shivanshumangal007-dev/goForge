package tensor

import "testing"

func TestStrides(t *testing.T) {
	data := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)

	if data.stride[0] != 3 && data.stride[1] != 1 {
		t.Fatalf("expected stride as [3,1] got %v", data.stride)
	}
}

func TestAt(t *testing.T) {
	data := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)

	if data.At(1, 2) != 6 {
		t.Fatalf("expected 3 got %v", data.At(1, 3))
	}
}
