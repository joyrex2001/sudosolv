package main

import (
	"fmt"

	"github.com/joyrex2001/sudosolv/internal/image"
	"github.com/joyrex2001/sudosolv/internal/mnist"
)

const network = "./internal/mnist/dataset/mnist-trained.bin"

func main() {
	img := image.NewPuzzleImage("_archive/IMG_6502.jpg")
	cel := img.GetSudokuCell(3, 2)
	// fmt.Printf("%v\n", cel)

	// mnist.Train(network, 1)

	inf, err := mnist.NewInference(network)
	if err != nil {
		fmt.Printf("error = %s\n", err)
		return
	}
	for i := 0; i < 3; i++ {
		res, err := inf.Predict(cel)
		if err != nil {
			fmt.Printf("error = %s\n", err)
		}
		fmt.Printf("res = %d\n", res)
	}

}
