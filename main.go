package main

import (
	"fmt"

	"github.com/joyrex2001/sudosolv/internal/image"
)

func main() {
	img := image.NewPuzzleImage("_archive/IMG_6502.jpg")
	cel := img.GetSudokuCell(3, 2)
	fmt.Printf("%v\n", cel)
}
