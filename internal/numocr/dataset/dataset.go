package dataset

import "gorgonia.org/tensor"

type Dataset interface {
	XY() (tensor.Tensor, tensor.Tensor, error)
	Epochs() int
	WeightsFile() string
}
