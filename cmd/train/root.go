package train

import (
	"github.com/spf13/cobra"
)

var Cmd = &cobra.Command{
	Use:   "train",
	Short: "Train a network for number recognition",
}

func init() {
	Cmd.PersistentFlags().StringP("weights", "w", "trained.bin", "Filename to write/update output weights")
	Cmd.PersistentFlags().Int("epochs", 5, "Number of training epochs")
}
