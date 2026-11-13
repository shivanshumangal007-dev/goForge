package main

import (
	"fmt"
	"goForge/autoGrad"
	"goForge/nn"
	"goForge/optim"
	"goForge/tensor"
	"math/rand"
)

func main() {
	rng := rand.New(rand.NewSource(42))

	model := nn.NewLinear(1, 1, rng)
	sg := optim.NewSgd(model.Params(), 0.05)

	var1 := autoGrad.NewVar(tensor.NewTensor([]float32{0, 1, 2, 3}, 4, 1))
	var2 := autoGrad.NewVar(tensor.NewTensor([]float32{1, 3, 5, 7}, 4, 1))

	for i := range 500 {
		pred := model.Forward(var1)
		diff := autoGrad.SubVar(pred, var2)
		loss := autoGrad.MeanVar(autoGrad.MulVar(diff, diff))
		sg.ZeroGrad()
		loss.Backward()
		//fmt.Println("grads:", model.W.Grad.Data, model.B.Grad.Data)
		sg.Step()

		if i%50 == 0 {
			fmt.Println(i, loss.Data.Data[0])
		}
	}

	fmt.Println("W:", model.W.Data.Data, "B:", model.B.Data.Data)
}
