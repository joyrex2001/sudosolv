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
	// Display(bl)

	wb := gocv.NewMat()
	gocv.Threshold(gr, &wb, 127, 255, 0)
	gocv.AdaptiveThreshold(gr, &wb, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinaryInv, 7, 1)
	// Display(wb)

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
	return crop
}

// GetSudokuCell will return 28x28 bytes for a given x,y of
// a cell within a sudoku puzzle.
func (pi *PuzzleImage) GetSudokuCell(x, y int) []byte {
	margin := int((width / 9) * 0.85)
	px := (width/9)*x + margin
	py := (width/9)*y + margin
	w := width/9 - 2*margin
	crop := pi.img.Region(image.Rect(px, py, px+w, py+w))

	res := gocv.NewMat()
	gocv.Resize(crop, &res, image.Point{28, 28}, 0, 0, gocv.InterpolationDefault)

	gr := gocv.NewMat()
	gocv.CvtColor(res, &gr, gocv.ColorBGRToGray)

	bl := gocv.NewMat()
	gocv.GaussianBlur(gr, &bl, image.Point{}, 5, 5, gocv.BorderDefault)

	wb := gocv.NewMat()
	gocv.Threshold(gr, &wb, 127, 255, 0)
	gocv.AdaptiveThreshold(gr, &wb, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinaryInv, 7, 8)
	// Display(wb)

	return wb.ToBytes()
}
