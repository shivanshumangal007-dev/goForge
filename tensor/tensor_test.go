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

func TestNewTensorPanic(t *testing.T) {

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()

	_ = NewTensor([]float32{1, 2, 3, 4, 5, 6}, 1, 3)

}

func TestStride3D(t *testing.T) {
	data := NewTensor([]float32{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, 2, 3, 4)
	want := []int{12, 4, 1}

	for i, v := range want {
		if data.stride[i] != v {
			t.Fatalf("expected %v got %v", v, data.stride[i])
		}
	}
}

func TestAdd(t *testing.T) {
	t1 := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	t2 := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)

	newT := Add(t1, t2)
	for i, v := range newT.data {
		if v != t1.data[i]+t2.data[i] {
			t.Fatalf("expected %v got %v", t1.data[i]+t2.data[i], v)
		}
	}
}

func TestAddShapeMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("AddShapeMismatch did not panic")
		}
	}()

	t1 := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	t2 := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 3, 2)
	Add(t1, t2)
}
