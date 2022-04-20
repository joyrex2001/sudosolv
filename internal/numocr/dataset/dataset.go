package dataset

import (
	"gorgonia.org/tensor"
)

// Dataset describes the interface that is used in the network for
// describing/providing datasets to be trained.
type Dataset interface {
	// XY should return two tensors, one for a 28x28 trainings dataset,
	// and one for the expected result as a tensor with a shape of 10.
	XY() (tensor.Tensor, tensor.Tensor, error)

	// Epochs should return the number of epochs that are required for
	// a properly trained network.
	Epochs() int

	// TestRatio should return the ratio as a float between 0 and 1
	// indicating how much of the data provided by XY can be used for
	// testing the network.
	TestRatio() float64
}
