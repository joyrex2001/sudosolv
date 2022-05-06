package image

import (
	"image"

	"gocv.io/x/gocv"
)

const (
	width     = 540 // easy to divide by 9
	threshold = 450 // 450 seems to work fine
)

// PuzzleImage is an object that represents a puzzle image.
type PuzzleImage struct {
	img gocv.Mat
}

// NewPuzzleImage will return a PuzzleImage object based on the
// given image file.
func NewPuzzleImage(file string) *PuzzleImage {
	img := gocv.IMRead(file, gocv.IMReadColor)
	return &PuzzleImage{img: getPuzzle(img)}
}

// GetPuzzle will return a OpenCV matrix with the biggest square of the
// given OpenCV matrix, assuming it's the Sudoku puzzle.
func getPuzzle(img gocv.Mat) gocv.Mat {
	gr := gocv.NewMat()
	gocv.CvtColor(img, &gr, gocv.ColorBGRToGray)

	bl := gocv.NewMat()
	gocv.GaussianBlur(gr, &bl, image.Point{}, 5, 5, gocv.BorderDefault)
	// display(bl)

	wb := gocv.NewMat()
	gocv.Threshold(gr, &wb, 127, 255, 0)
	gocv.AdaptiveThreshold(gr, &wb, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinaryInv, 7, 1)
	// display(wb)

	cn := gocv.FindContours(wb, gocv.RetrievalList, gocv.ChainApproxSimple)
	box := gocv.NewPointVector()
	for i := 0; i < cn.Size(); i++ {
		apx := gocv.ApproxPolyDP(cn.At(i), threshold, true)
		if isSquare(apx) && isBigger(apx, box) {
			box = sortCoords(apx)
		}
	}

	// bx := gocv.NewPointsVector()
	// bx.Append(box)
	// logPointVector(box)
	// gocv.DrawContours(&img, bx, -1, color.RGBA{0, 255, 0, 0}, 10)
	// Display(img)

	tb := gocv.NewPointVector()
	tb.Append(image.Point{0, 0})
	tb.Append(image.Point{0, width})
	tb.Append(image.Point{width, width})
	tb.Append(image.Point{width, 0})
	pt := gocv.GetPerspectiveTransform(box, tb)

	crop := gocv.NewMat()
	gocv.WarpPerspective(img, &crop, pt, image.Point{width, width})
	// display(crop)

	return crop
}

// Display will display the puzzle data and wait until a key has
// been pressed.
func (pi *PuzzleImage) Display() {
	display(pi.img)
}

// getSudokuCellArea will return a rectangle with coordinates in
// which the given sudoko cell is present within the image.
func getSudokuCellArea(x, y int) image.Rectangle {
	margin := int((width / 9) * 0.1)
	px := (width/9)*x + margin
	py := (width/9)*y + margin
	w := width/9 - 2*margin
	return image.Rect(px, py, px+w, py+w)
}

// GetSudokuCell will return 28x28 bytes for a given x,y of
// a cell within a sudoku puzzle.
func (pi *PuzzleImage) GetSudokuCell(x, y int) []byte {
	area := getSudokuCellArea(x, y)
	crop := pi.img.Region(area)

	// crop number
	gr := gocv.NewMat()
	thres := gocv.NewMat()
	gocv.CvtColor(crop, &gr, gocv.ColorBGRToGray)
	gocv.AdaptiveThreshold(gr, &thres, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinaryInv, 7, 8)
	bx := biggestBoundingBox(thres)
	if bx.Size().X*bx.Size().Y > 10 {
		// fmt.Printf("bx=%v\n", bx)
		// bx = addMargin(bx, 1)
		crop = crop.Region(bx)
	}

	// convert to hard black and white
	wb := gocv.NewMat()
	gocv.CvtColor(crop, &gr, gocv.ColorBGRToGray)
	gocv.AdaptiveThreshold(gr, &wb, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinaryInv, 31, 10)

	// crop to 28x28
	res := gocv.NewMat()
	gocv.Resize(wb, &res, image.Point{28, 28}, 0, 0, gocv.InterpolationDefault)
	// display(res)

	return res.ToBytes()
}
