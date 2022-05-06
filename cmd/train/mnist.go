package train

import (
	"fmt"

	"github.com/joyrex2001/sudosolv/internal/numocr/dataset/mnist"
	"github.com/joyrex2001/sudosolv/internal/numocr/network"

	"github.com/spf13/cobra"
)

var mnistCmd = &cobra.Command{
	Use:   "mnist",
	Short: "Train a mnist network for number recognition using the mnist dataset.",
	Run:   trainMnist,
}

func init() {
	Cmd.AddCommand(mnistCmd)
	mnistCmd.Flags().String("dataloc", "", "Path where the mnist dataset files are located")
	Cmd.MarkFlagRequired("dataset")
}

func trainMnist(cmd *cobra.Command, args []string) {
	weights, _ := cmd.Flags().GetString("weights")
	epochs, _ := cmd.Flags().GetInt("epochs")

	dataloc, _ := cmd.Flags().GetString("dataloc")
	dataset := mnist.NewMnistDataset(dataloc, epochs)

	if err := network.Train(weights, dataset); err != nil {
		fmt.Printf("error = %s\n", err)
		return
	}
}
