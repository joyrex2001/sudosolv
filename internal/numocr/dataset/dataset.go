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

	// WeightsFile should return the path of the file in which the
	// trained weights should be stored during training, or loaded
	// during inference.
	WeightsFile() string
}
