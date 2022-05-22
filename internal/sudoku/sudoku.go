package sudoku

import (
	"fmt"

	"github.com/joyrex2001/sudosolv/internal/image"
	"github.com/joyrex2001/sudosolv/internal/numocr/classifier"
)

// Sudoku describes a sudoku puzzle object.
type Sudoku struct {
	Cell [][]int `json:"cell"`
}

// NewSudokuFromPuzzleImage will instantiate a new Sudoku object given a
// PuzzleImage object and a numocr classifier Inference object to decode
// the image.
func NewSudokuFromPuzzleImage(img *image.PuzzleImage, inf *classifier.Inference) (*Sudoku, error) {
	cell := make([][]int, 9)
	for x := 0; x < 9; x++ {
		cell[x] = make([]int, 9)
		for y := 0; y < 9; y++ {
			cel := img.GetSudokuCell(x, y)
			res, _, err := inf.Predict(cel)
			if err != nil {
				return nil, err
			}
			if res < 1 {
				res = 0
			}
			cell[x][y] = res
		}
	}
	return &Sudoku{Cell: cell}, nil
}

// IsValid validates if the sudoku is valid.
func (sd *Sudoku) IsValid() bool {
	// vertical check
	for x := 0; x < 9; x++ {
		done := map[int]bool{}
		for y := 0; y < 9; y++ {
			n := sd.Cell[x][y]
			if n < 1 {
				continue
			}
			if _, ok := done[n]; ok {
				return false
			}
			done[n] = true
		}
	}
	// horizontal check
	for x := 0; x < 9; x++ {
		done := map[int]bool{}
		for y := 0; y < 9; y++ {
			n := sd.Cell[x][y]
			if n < 1 {
				continue
			}
			if _, ok := done[n]; ok {
				return false
			}
			done[n] = true
		}
	}
	// block check
	for bx := 0; bx < 3; bx++ {
		for by := 0; by < 3; by++ {
			done := map[int]bool{}
			for c := 0; c < 9; c++ {
				n := sd.Cell[bx*3+c/3][by*3+c%3]
				if n < 1 {
					continue
				}
				if _, ok := done[n]; ok {
					return false
				}
				done[n] = true
			}
		}
	}
	return true
}

// IsCompleted will check if all cells in the sudoku are filled in.
func (sd *Sudoku) IsCompleted() bool {
	for x := 0; x < 9; x++ {
		for y := 0; y < 9; y++ {
			if sd.Cell[x][y] < 1 {
				return false
			}
		}
	}
	return true
}

// String will return a printable version of the Sudoku object.
func (sd *Sudoku) String() string {
	out := ""
	for y := 0; y < 9; y++ {
		if y != 0 && y%3 == 0 {
			out += "-----------+-----------+-----------\n"
		}
		for x := 0; x < 9; x++ {
			if x != 0 && x%3 == 0 {
				out += "  |"
			}
			res := sd.Cell[x][y]
			if res < 1 {
				out += "   "
			} else {
				out += fmt.Sprintf("%3d", res)
			}
		}
		out += "  \n"
	}
	return out
}

// Solve will use backtracking to provide a solution for the sudoku.
func (sd *Sudoku) Solve() bool {
	if sd.IsCompleted() {
		return true
	}
	for x := 0; x < 9; x++ {
		for y := 0; y < 9; y++ {
			if sd.Cell[x][y] == 0 {
				for n := 1; n <= 9; n++ {
					sd.Cell[x][y] = n
					if sd.IsValid() {
						if sd.Solve() {
							return true
						}
						sd.Cell[x][y] = 0
					} else {
						sd.Cell[x][y] = 0
					}
				}
				return false
			}
		}
	}
	return false
}
