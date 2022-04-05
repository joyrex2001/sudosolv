package main

import (
	"fmt"

	"github.com/joyrex2001/sudosolv/internal/image"
	"github.com/joyrex2001/sudosolv/internal/numocr/dataset/mnist"
	"github.com/joyrex2001/sudosolv/internal/numocr/network"
)

func main() {
	dataset := mnist.NewMnistDataset()

	if err := network.Train(dataset); err != nil {
		fmt.Printf("error = %s\n", err)
		return
	}

	inf, err := network.NewInference(dataset)
	if err != nil {
		fmt.Printf("error = %s\n", err)
		return
	}

	img := image.NewPuzzleImage("_archive/IMG_6501.jpg")
	cel := img.GetSudokuCell(3, 2)
	// fmt.Printf("%v\n", cel)
	res, err := inf.Predict(cel)
	if err != nil {
		fmt.Printf("error = %s\n", err)
		return
	}
	fmt.Printf("%3d\n", res)
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
			res, err := inf.Predict(cel)
			if err != nil {
				fmt.Printf("error = %s\n", err)
			}
			fmt.Printf("%3d", res)
		}
		fmt.Printf("\n")
	}

}
