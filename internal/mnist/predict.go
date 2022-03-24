package mnist

import (
	"fmt"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Inference is the internal representation of the inference object.
type Inference struct {
	vm gorgonia.VM
	x  *gorgonia.Node
	g  *gorgonia.ExprGraph
	m  *convnet
}

// NewInference will return a new inference object to handle
// predictions.
func NewInference() (*Inference, error) {
	in := &Inference{}
	in.g = gorgonia.NewGraph()
	in.x = gorgonia.NewTensor(in.g, tensor.Float64, 4, gorgonia.WithShape(1, 1, 28, 28))
	in.m = newConvNet(in.g)

	if err := in.m.load(backup); err != nil {
		return nil, fmt.Errorf("Failed loading network: %v", err)
	}

	if err := in.m.fwd(in.x); err != nil {
		return nil, fmt.Errorf("Failed: %v", err)
	}

	in.vm = gorgonia.NewTapeMachine(in.g)

	return in, nil
}

// Predict will take a 28x28 image and return the integer it
// thinks best matches the image. If no match is found, it will
// return -1.
func (in *Inference) Predict(image []byte) (int, error) {
	defer in.vm.Reset()

	xVal := prepareX([]RawImage{image}, tensor.Float64)
	if err := xVal.(*tensor.Dense).Reshape(1, 1, 28, 28); err != nil {
		return -1, fmt.Errorf("Unable to reshape: %v", err)
	}

	gorgonia.Let(in.x, xVal)
	if err := in.vm.RunAll(); err != nil {
		return -1, fmt.Errorf("Failed running expression: %v", err)
	}

	outs := in.m.predVal.Data().([]float64)
	res := -1
	ms := 0.
	for n, s := range outs {
		if s > 0.8 && s > ms {
			res = n
			ms = s
		}
	}

	// fmt.Printf("digit: %v\n", xVal.Data().([]float64))
	fmt.Printf("outs: %v\n", outs)

	return res, nil
}
