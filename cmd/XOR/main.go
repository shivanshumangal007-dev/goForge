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

	model := nn.NewMLP(2, 8, 1, rng)
	sg := optim.NewSgd(model.Params(), 0.1)
	x := autoGrad.NewVar(tensor.NewTensor([]float32{0, 0, 0, 1, 1, 0, 1, 1}, 4, 2))
	y := autoGrad.NewVar(tensor.NewTensor([]float32{0, 1, 1, 0}, 4, 1))

	for i := range 2000 {
		pred := model.Forward(x)
		diff := autoGrad.SubVar(pred, y)
		loss := autoGrad.MeanVar(autoGrad.MulVar(diff, diff))
		sg.ZeroGrad()
		loss.Backward()
		//fmt.Println("grads:", model.W.Grad.Data, model.B.Grad.Data)
		sg.Step()

		if i%200 == 0 {
			fmt.Println(i, loss.Data.Data[0])
		}
	}

	perd1 := model.Forward(x)
	for i := range perd1.Data.Data {
		fmt.Println(i, " ==> ", perd1.Data.Data[i])
	}
	//fmt.Println("W:", model.W.Data.Data, "B:", model.B.Data.Data)
}
