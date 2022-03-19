package main

import (
	"github.com/joyrex2001/sudosolv/internal/image"

	"gocv.io/x/gocv"
)

func main() {
	img := gocv.IMRead("_archive/IMG_6502.jpg", gocv.IMReadColor)
	defer img.Close()

	puzzle := image.GetPuzzle(img)
	image.Display(puzzle)

	cel := image.GetSudokuCell(puzzle, 3, 2)
	image.Display(cel)
}
