package main

import (
	"fmt"

	"github.com/joyrex2001/sudosolv/internal/image"
	"github.com/joyrex2001/sudosolv/internal/numocr/dataset/generated"
	"github.com/joyrex2001/sudosolv/internal/numocr/network"
)

func main() {
	// dataset := mnist.NewMnistDataset()
	// if err := network.Train(dataset); err != nil {
	// 	fmt.Printf("error = %s\n", err)
	// 	return
	// }

	for _, f := range generated.Fonts {
		fmt.Printf("training %s\n", f)
		d := generated.NewGeneratedDatasetForFont(f)
		if err := network.Train(d); err != nil {
			fmt.Printf("error while training %s = %s\n", f, err)
		}
	}

	for _, f := range generated.Fonts {
		dataset := generated.NewGeneratedDatasetForFont(f)

		inf, err := network.NewInference(dataset)
		if err != nil {
			fmt.Printf("error = %s\n", err)
			return
		}

		img := image.NewPuzzleImage("_archive/IMG_6501.jpg")
		cel := img.GetSudokuCell(3, 2)
		// fmt.Printf("%v\n", cel)
		res, acc, err := inf.Predict(cel)
		if err != nil {
			fmt.Printf("error = %s\n", err)
			return
		}
		fmt.Printf("%3d (%f)\n", res, acc)
		// return

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
				fmt.Printf("%3d", res)
			}
			fmt.Printf("\n")
		}
	}
}
