package classifier

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

	if err := in.nn.fwd(in.x, false); err != nil {
		return nil, fmt.Errorf("failed: %v", err)
	}

	if err := in.nn.load(weights); err != nil {
		return nil, fmt.Errorf("failed loading weights: %v", err)
	}

	in.vm = gorgonia.NewTapeMachine(in.g)

	return in, nil
}

// Predict will take a 28x28 image and return the integer it
// thinks best matches the image. If no match is found, it will
// return -1.
func (in *Inference) Predict(image []byte) (int, float64, error) {
	in.vm.Reset()

	if isBlankImage(image) {
		return -1, 0, nil
	}

	x := imageTensor(image)
	if err := x.(*tensor.Dense).Reshape(1, 1, 28, 28); err != nil {
		return -1, 0, fmt.Errorf("unable to reshape: %s", err)
	}

	if err := gorgonia.Let(in.x, x); err != nil {
		return -1, 0, fmt.Errorf("error setting inputs: %s", err)
	}

	if err := in.vm.RunAll(); err != nil {
		return -1, 0, fmt.Errorf("failed running expression: %s", err)
	}

	y, err := in.nn.output()
	if err != nil {
		return -1, 0, fmt.Errorf("unable predict: %s", err)
	}

	res := -1
	ms := float64(0.)
	for n, s := range y {
		if s > 0.9 && s > ms {
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
	x := make([]float64, len(M))
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

// isBlankImage will check if the given image is a blank image and
// doesn't seem to contain a number.
func isBlankImage(image []byte) bool {
	sum := 0
	for _, h := range image {
		sum += int(h)
	}
	return sum < 30*255 // less than 30 intense bright pixels within the 28x28
}
