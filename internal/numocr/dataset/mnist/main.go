package mnist

import (
	"github.com/joyrex2001/sudosolv/internal/numocr/dataset"
	"gorgonia.org/tensor"
)

const (
	datatype = "train" // valid options are "train" or "test"
	dataloc  = "./internal/numocr/dataset/mnist/dataset/"
)

// MnistDataset is the object that describes the mnist dataset.
type MnistDataset struct {
	filename string
	epochs   int
}

// NewMnistDataset wil create a new MnistDataset instance.
func NewMnistDataset() dataset.Dataset {
	return &MnistDataset{
		epochs:   1,
		filename: "./internal/numocr/dataset/mnist/trained.bin",
	}
}

// WeightsFile will return the filename of the stored weights.
func (fd *MnistDataset) WeightsFile() string {
	return fd.filename
}

// Epochs returns the number of epochs that should run during training.
func (fd *MnistDataset) Epochs() int {
	return fd.epochs
}

// XY returns the data to be trained.
func (fd *MnistDataset) XY() (tensor.Tensor, tensor.Tensor, error) {
	return loadMnist(datatype, dataloc, tensor.Float64)
}

// TestRatio returns how much can of the dataset can be used for
// validating the network.
func (fd *MnistDataset) TestRatio() float64 {
	return 0.1
}
