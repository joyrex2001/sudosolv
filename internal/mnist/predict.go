package mnist

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func Predict(image []byte) int {
	g := gorgonia.NewGraph()
	x := gorgonia.NewTensor(g, tensor.Float64, 4, gorgonia.WithShape(1, 1, 28, 28), gorgonia.WithName("x"))
	m := newConvNet(g)

	if err := m.load(backup); err != nil {
		log.Fatalf("Failed: %v", err)
	}

	if err := m.fwd(x); err != nil {
		log.Fatalf("Failed: %v", err)
	}

	vm := gorgonia.NewTapeMachine(g)
	defer vm.Close()

	xVal := prepareX([]RawImage{image}, tensor.Float64)
	if err := xVal.(*tensor.Dense).Reshape(1, 1, 28, 28); err != nil {
		log.Fatalf("Unable to reshape %v", err)
	}

	gorgonia.Let(x, xVal)
	if err := vm.RunAll(); err != nil {
		log.Fatalf("Failed: %v", err)
	}
	outs := m.predVal.Data().([]float64)
	vm.Reset()

	fmt.Printf("outs: %v\n", outs)

	res := -1
	ms := 0.
	for n, s := range outs {
		if s > 0.8 && s > ms {
			res = n
			ms = s
		}
	}

	return res
}
