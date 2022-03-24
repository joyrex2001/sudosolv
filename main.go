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
	// mnist.Train()
	res := mnist.Predict(cel)
	fmt.Printf("res = %d\n", res)
}
