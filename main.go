package main

import (
	"fmt"

	"github.com/joyrex2001/sudosolv/internal/image"
	"github.com/joyrex2001/sudosolv/internal/mnist"
)

func main() {
	img := image.NewPuzzleImage("_archive/IMG_6502.jpg")
	cel := img.GetSudokuCell(3, 2)
	fmt.Printf("%v\n", cel)

	mnist.Train()

	inf, err := mnist.NewInference()
	if err != nil {
		fmt.Printf("error = %s\n", err)
		return
	}
	res, err := inf.Predict(cel)
	if err != nil {
		fmt.Printf("error = %s\n", err)
	}

	fmt.Printf("res = %d\n", res)
}
