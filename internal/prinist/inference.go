package prinist

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
	nn *network
}

// NewInference will return a new inference object with the given
// trained weights file to handle predictions.
func NewInference(weights string) (*Inference, error) {
	in := &Inference{}
	in.g = gorgonia.NewGraph()
	in.x = gorgonia.NewTensor(in.g, tensor.Float64, 4, gorgonia.WithShape(1, 1, 28, 28), gorgonia.WithName("x"))
	in.nn = newNetwork(in.g)
	in.nn.disableDropOut()

	if err := in.nn.fwd(in.x); err != nil {
		return nil, fmt.Errorf("Failed: %v", err)
	}

	if err := in.nn.load(weights); err != nil {
		return nil, fmt.Errorf("Failed loading weights: %v", err)
	}

	in.vm = gorgonia.NewTapeMachine(in.g)

	return in, nil
}

// Predict will take a 28x28 image and return the integer it
// thinks best matches the image. If no match is found, it will
// return -1.
func (in *Inference) Predict(image []byte) (int, error) {
	in.vm.Reset()

	x := X2Tensor(image)
	if err := x.(*tensor.Dense).Reshape(1, 1, 28, 28); err != nil {
		return -1, fmt.Errorf("Unable to reshape: %s", err)
	}

	if err := gorgonia.Let(in.x, x); err != nil {
		return -1, fmt.Errorf("Error setting inputs: %s", err)
	}

	if err := in.vm.RunAll(); err != nil {
		return -1, fmt.Errorf("Failed running expression: %s", err)
	}

	y, err := in.nn.output()
	if err != nil {
		return -1, fmt.Errorf("Unable predict: %s", err)
	}

	res := -1
	ms := float64(0.)
	for n, s := range y {
		if s > 0.8 && s > ms {
			res = n
			ms = s
		}
	}

	// fmt.Printf("outs = %v\n", y)

	return res, nil
}
