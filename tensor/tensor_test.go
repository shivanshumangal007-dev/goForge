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
	td := Transpose(data)

	if td.At(2, 1) != 6 {
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
func TestMultiply(t *testing.T) {
	t1 := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 3, 2)
	i := NewTensor([]float32{1, 0, 0, 1}, 2, 2)

	m := Multiply(t1, i)

	for i, v := range m.data {
		if v != t1.data[i] {
			t.Fatalf("expected %f got %f", t1.data[i], v)
		}
	}
}

func TestMultiplyShapeMismatch(t *testing.T) {
	t1 := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	i := NewTensor([]float32{1, 0, 0, 1}, 2, 2)

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("MultiplyShapeMismatch did not panic")
		}
	}()

	Multiply(t1, i)
}
func TestMultiplyWithTranspose(t *testing.T) {
	a := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	got := Multiply(a, Transpose(a)) // [2,3] x [3,2] = [2,2]

	want := []float32{14, 32, 32, 77}
	for i := range want {
		if got.data[i] != want[i] {
			t.Fatalf("got %v, want %v", got.data, want)
		}
	}
}
