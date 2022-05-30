package image

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"os"

	"gocv.io/x/gocv"
)

const (
	width = 540 // easy to divide by 9
)

// PuzzleImage is an object that represents a puzzle image.
type PuzzleImage struct {
	img gocv.Mat
}

// NewPuzzleImage will return a PuzzleImage object based on the
// given image file.
func NewPuzzleImage(file string) (*PuzzleImage, error) {
	if _, err := os.Stat(file); err != nil {
		return nil, err
	}
	img := gocv.IMRead(file, gocv.IMReadColor)
	defer img.Close()
	pimg, err := getPuzzle(img)
	if err != nil {
		return nil, err
	}
	return &PuzzleImage{img: pimg}, nil
}

// NewPuzzleImageFromReader will return a PuzzleImage object based
// on the given raw image data.
func NewPuzzleImageFromReader(r io.ReadSeeker) (*PuzzleImage, error) {
	imgraw, _, err := image.Decode(r)
	if err != nil {
		return nil, err
	}
	img, err := gocv.ImageToMatRGBA(imgraw)
	if err != nil {
		return nil, fmt.Errorf("error processing image: %s", err)
	}
	r.Seek(0, io.SeekStart)
	img = fixOrientation(img, getOrientation(r))
	pimg, err := getPuzzle(img)
	if err != nil {
		return nil, err
	}
	return &PuzzleImage{img: pimg}, nil
}

// GetPuzzle will return a OpenCV matrix with the biggest square of the
// given OpenCV matrix, assuming it's the Sudoku puzzle.
func getPuzzle(img gocv.Mat) (gocv.Mat, error) {
	threshold := float64(img.Rows() / 3) // at least 1/3 of the image height
	return getBiggestBox(img, threshold)
}

// getBiggestBox will return a OpenCV matrix containing the biggest square
// which also matches the given threshold.
func getBiggestBox(img gocv.Mat, threshold float64) (gocv.Mat, error) {
	gr := gocv.NewMat()
	defer gr.Close()
	gocv.CvtColor(img, &gr, gocv.ColorBGRToGray)

	bl := gocv.NewMat()
	defer bl.Close()
	gocv.GaussianBlur(gr, &bl, image.Point{}, 5, 5, gocv.BorderDefault)
	// display(bl)

	wb := gocv.NewMat()
	defer wb.Close()
	gocv.Threshold(gr, &wb, 127, 255, 0)
	gocv.AdaptiveThreshold(gr, &wb, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinaryInv, 9, 1)
	// display(wb)

	cn := gocv.FindContours(wb, gocv.RetrievalList, gocv.ChainApproxSimple)
	box := gocv.NewPointVector()
	for i := 0; i < cn.Size(); i++ {
		apx := gocv.ApproxPolyDP(cn.At(i), threshold, true)
		if isSquare(apx) && isBigger(apx, box) {
			box = sortCoords(apx)
		}
	}

	if box.Size() != 4 {
		return img, fmt.Errorf("could not detect a sudoku puzzle")
	}

	// bx := gocv.NewPointsVector()
	// bx.Append(box)
	// logPointVector(box)
	// gocv.DrawContours(&img, bx, -1, color.RGBA{0, 255, 0, 0}, 10)
	// display(img)

	tb := gocv.NewPointVector()
	tb.Append(image.Point{0, 0})
	tb.Append(image.Point{0, width})
	tb.Append(image.Point{width, width})
	tb.Append(image.Point{width, 0})
	pt := gocv.GetPerspectiveTransform(box, tb)

	crop := gocv.NewMat()
	gocv.WarpPerspective(img, &crop, pt, image.Point{width, width})
	// display(crop)

	return crop, nil
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
	// display(crop)

	// crop number
	gr := gocv.NewMat()
	defer gr.Close()
	thres := gocv.NewMat()
	defer thres.Close()
	gocv.CvtColor(crop, &gr, gocv.ColorBGRToGray)
	gocv.AdaptiveThreshold(gr, &thres, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinaryInv, 9, 1)
	// display(thres)
	bx := biggestBoundingBox(thres)
	if bx.Size().X < 20 && bx.Size().Y < 20 {
		return make([]byte, 28*28)
	}

	// fmt.Fprintf(os.Stderr, "bx=%v\n", bx)
	// bx = addMargin(bx, 1)
	crop = crop.Region(bx)

	// convert to hard black and white
	wb := gocv.NewMat()
	defer wb.Close()
	gocv.CvtColor(crop, &gr, gocv.ColorBGRToGray)
	gocv.AdaptiveThreshold(gr, &wb, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinaryInv, 31, 10)

	// crop to 28x28
	res := gocv.NewMat()
	gocv.Resize(wb, &res, image.Point{28, 28}, 0, 0, gocv.InterpolationDefault)
	// display(res)

	return res.ToBytes()
}
