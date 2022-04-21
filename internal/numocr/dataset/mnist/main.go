package mnist

import (
	"github.com/joyrex2001/sudosolv/internal/numocr/dataset"
	"gorgonia.org/tensor"
)

const (
	datatype = "train" // valid options are "train" or "test"
)

// MnistDataset is the object that describes the mnist dataset.
type MnistDataset struct {
	epochs  int
	dataloc string
}

// NewMnistDataset wil create a new MnistDataset instance.
func NewMnistDataset(dataloc string, epochs int) dataset.Dataset {
	return &MnistDataset{
		epochs:  epochs,
		dataloc: dataloc,
	}
}

// Epochs returns the number of epochs that should run during training.
func (fd *MnistDataset) Epochs() int {
	return fd.epochs
}

// XY returns the data to be trained.
func (fd *MnistDataset) XY() (tensor.Tensor, tensor.Tensor, error) {
	return loadMnist(datatype, fd.dataloc, tensor.Float64)
}

// TestRatio returns how much can of the dataset can be used for
// validating the network.
func (fd *MnistDataset) TestRatio() float64 {
	return 0.1
}
