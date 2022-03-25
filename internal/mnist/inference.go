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
	m  *network
}

// NewInference will return a new inference object with the given
// trained network file to handle predictions.
func NewInference(network string) (*Inference, error) {
	in := &Inference{}
	in.g = gorgonia.NewGraph()
	in.x = gorgonia.NewTensor(in.g, tensor.Float64, 4, gorgonia.WithShape(1, 1, 28, 28))
	in.m = newNetwork(in.g)
	in.m.disableDropOut()

	if err := in.m.load(network); err != nil {
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
	in.vm.Reset()

	x := prepareX([]RawImage{image}, tensor.Float64)
	if err := x.(*tensor.Dense).Reshape(1, 1, 28, 28); err != nil {
		return -1, fmt.Errorf("Unable to reshape: %s", err)
	}

	gorgonia.Let(in.x, x)
	if err := in.vm.RunAll(); err != nil {
		return -1, fmt.Errorf("Failed running expression: %s", err)
	}

	y, err := in.m.output()
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

	// fmt.Printf("digit: %v\n", x.Data().([]float64))
	// fmt.Printf("outs: %v\n\n\n", y)

	return res, nil
}
