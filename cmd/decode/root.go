package decode

import (
	"fmt"

	"github.com/joyrex2001/sudosolv/internal/image"
	"github.com/joyrex2001/sudosolv/internal/numocr/classifier"

	"github.com/spf13/cobra"
)

var Cmd = &cobra.Command{
	Use:   "decode",
	Short: "Decode the sudoku to text for given image.",
	Run:   decodeImage,
}

func init() {
	Cmd.Flags().StringP("weights", "w", "trained.bin", "Filename to write/update output weights")
	Cmd.Flags().StringP("file", "f", "", "Filename of a picture of a sudoko to be decoded")
	Cmd.Flags().Bool("display", false, "Display the cropped sudoku image and wait until a key press")
	Cmd.MarkFlagRequired("file")
}

func decodeImage(cmd *cobra.Command, args []string) {
	weights, _ := cmd.Flags().GetString("weights")
	file, _ := cmd.Flags().GetString("file")
	display, _ := cmd.Flags().GetBool("display")

	inf, err := classifier.NewInference(weights)
	if err != nil {
		fmt.Printf("error = %s\n", err)
		return
	}

	img := image.NewPuzzleImage(file)
	if display {
		img.Display()
		return
	}

	for y := 0; y < 9; y++ {
		if y != 0 && y%3 == 0 {
			fmt.Printf("-----------+-----------+-----------\n")
		}
		for x := 0; x < 9; x++ {
			if x != 0 && x%3 == 0 {
				fmt.Printf("  |")
			}
			cel := img.GetSudokuCell(x, y)
			res, _, err := inf.Predict(cel)
			if err != nil {
				fmt.Printf("error = %s\n", err)
			}
			if res < 1 {
				fmt.Printf("   ")
			} else {
				fmt.Printf("%3d", res)
			}
		}
		fmt.Printf("\n")
	}
}
