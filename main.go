package main

import (
	"fmt"

	"github.com/joyrex2001/sudosolv/internal/image"
	"github.com/joyrex2001/sudosolv/internal/mnist"
)

const (
	network = "./internal/mnist/dataset/mnist-trained.bin"
	epochs  = 1
)

func main() {
	// if err := mnist.Train(network, epochs); err != nil {
	// 	fmt.Printf("error = %s\n", err)
	// 	return
	// }

	inf, err := mnist.NewInference(network)
	if err != nil {
		fmt.Printf("error = %s\n", err)
		return
	}

	img := image.NewPuzzleImage("_archive/IMG_6502.jpg")
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
			fmt.Printf("%3.d", res)
		}
		fmt.Printf("\n")
	}

}
