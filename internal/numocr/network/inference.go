package network

import (
	"fmt"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	"github.com/joyrex2001/sudosolv/internal/numocr/dataset"
)

// Inference is the internal representation of the inference object.
type Inference struct {
	vm      gorgonia.VM
	x       *gorgonia.Node
	g       *gorgonia.ExprGraph
	nn      *network
	dataset dataset.Dataset
}

// NewInference will return a new inference object with the given
// trained weights file to handle predictions.
func NewInference(dataset dataset.Dataset) (*Inference, error) {
	in := &Inference{
		dataset: dataset,
	}
	in.g = gorgonia.NewGraph()
	in.x = gorgonia.NewTensor(in.g, tensor.Float64, 4, gorgonia.WithShape(1, 1, 28, 28), gorgonia.WithName("x"))
	in.nn = newNetwork(in.g)
	in.nn.disableDropOut()

	if err := in.nn.fwd(in.x); err != nil {
		return nil, fmt.Errorf("Failed: %v", err)
	}

	if err := in.nn.load(dataset.WeightsFile()); err != nil {
		return nil, fmt.Errorf("Failed loading weights: %v", err)
	}

	in.vm = gorgonia.NewTapeMachine(in.g)

	return in, nil
}

// Predict will take a 28x28 image and return the integer it
// thinks best matches the image. If no match is found, it will
// return -1.
func (in *Inference) Predict(image []byte) (int, float64, error) {
	in.vm.Reset()

	x := imageTensor(image)
	if err := x.(*tensor.Dense).Reshape(1, 1, 28, 28); err != nil {
		return -1, 0, fmt.Errorf("Unable to reshape: %s", err)
	}

	if err := gorgonia.Let(in.x, x); err != nil {
		return -1, 0, fmt.Errorf("Error setting inputs: %s", err)
	}

	if err := in.vm.RunAll(); err != nil {
		return -1, 0, fmt.Errorf("Failed running expression: %s", err)
	}

	y, err := in.nn.output()
	if err != nil {
		return -1, 0, fmt.Errorf("Unable predict: %s", err)
	}

	res := -1
	ms := float64(0.)
	for n, s := range y {
		if s > ms {
			res = n
			ms = s
		}
	}

	// fmt.Printf("outs = %v\n", y)

	return res, ms, nil
}

// imageTensor converts a given []byte to a tensor with floats to use as
// input for the network.
func imageTensor(M []byte) tensor.Tensor {
	cols := 28 * 28
	rows := len(M) / cols
	x := make([]float64, len(M), len(M))
	for i, px := range M {
		max := 255.
		n := float64(px)/max*0.9 + 0.1
		if n == 1.0 {
			n = 0.999
		}
		x[i] = n
	}
	return tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(x))
}
