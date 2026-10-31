package tensor

import "testing"

func TestStrides(t *testing.T) {
	Data := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)

	if Data.stride[0] != 3 && Data.stride[1] != 1 {
		t.Fatalf("expected stride as [3,1] got %v", Data.stride)
	}
}

func TestAt(t *testing.T) {
	Data := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	td := Transpose(Data)

	if td.At(2, 1) != 6 {
		t.Fatalf("expected 3 got %v", Data.At(1, 3))
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
	Data := NewTensor([]float32{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, 2, 3, 4)
	want := []int{12, 4, 1}

	for i, v := range want {
		if Data.stride[i] != v {
			t.Fatalf("expected %v got %v", v, Data.stride[i])
		}
	}
}

func TestAdd(t *testing.T) {
	t1 := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	t2 := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)

	newT := Add(t1, t2)
	for i, v := range newT.Data {
		if v != t1.Data[i]+t2.Data[i] {
			t.Fatalf("expected %v got %v", t1.Data[i]+t2.Data[i], v)
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

	for i, v := range m.Data {
		if v != t1.Data[i] {
			t.Fatalf("expected %f got %f", t1.Data[i], v)
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
		if got.Data[i] != want[i] {
			t.Fatalf("got %v, want %v", got.Data, want)
		}
	}
}

func TestNewShapeError(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic because shape will not match")
		}
	}()
	a := NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	ReShape(a, 3, 1)
}

func TestZeroTensor(t *testing.T) {
	ZeroArr := Zeros(2, 3)

	for _, v := range ZeroArr.Data {
		if v != 0 {
			t.Fatalf("expected 0 got %v", v)
		}
	}
}
