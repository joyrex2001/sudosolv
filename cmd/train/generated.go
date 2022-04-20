package train

import (
	"fmt"

	"github.com/joyrex2001/sudosolv/internal/numocr/dataset/generated"
	"github.com/joyrex2001/sudosolv/internal/numocr/network"

	"github.com/spf13/cobra"
)

var generatedCmd = &cobra.Command{
	Use:   "generated",
	Short: "Train a network for number recognition using a generated dataset.",
	Run:   trainGenerated,
}

func init() {
	Cmd.AddCommand(generatedCmd)
	generatedCmd.Flags().Int("size", 60000, "Size of the dataset to be generated")
	generatedCmd.Flags().Bool("noise", false, "Add noise to the dataset")
	generatedCmd.Flags().Bool("rndsize", false, "Randomly vary the size of the fonts")
}

func trainGenerated(cmd *cobra.Command, args []string) {
	weights, _ := cmd.Flags().GetString("weights")
	dataset := generated.NewGeneratedDataset()
	if err := network.Train(weights, dataset); err != nil {
		fmt.Printf("error = %s\n", err)
		return
	}
}
