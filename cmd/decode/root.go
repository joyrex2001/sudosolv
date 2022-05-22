package decode

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/joyrex2001/sudosolv/internal/image"
	"github.com/joyrex2001/sudosolv/internal/numocr/classifier"
	"github.com/joyrex2001/sudosolv/internal/sudoku"
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
		fmt.Printf("error instantiating classifier: %s\n", err)
		return
	}

	img, _ := image.NewPuzzleImage(file)
	if display {
		img.Display()
		return
	}

	sd, err := sudoku.NewSudokuFromPuzzleImage(img, inf)
	if err != nil {
		fmt.Printf("error decoding sudoku: %s\n", err)
		return
	}
	if sd.IsValid() {
		sd.Solve()
		fmt.Printf("%s", sd)
	} else {
		fmt.Printf("sudoku is invalid!\n\n")
		fmt.Printf("%s", sd)
	}
}
