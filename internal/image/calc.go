package image

import (
	"image"

	"gocv.io/x/gocv"
)

// max will return the maximum value of the given integers.
func max(x ...int) int {
	max := 0
	for _, n := range x {
		if n > max {
			max = n
		}
	}
	return max
}

// sideLength will given a length indicator of the side as
// described with the two given coordinates.
func sideLength(a, b image.Point) int {
	return (a.X-b.X)*(a.X-b.X) + (a.Y-b.Y)*(a.Y-b.Y)
}

// isSquare will return true if the given OpenCV PointVector
// can be considered a (warped) square.
func isSquare(pv gocv.PointVector) bool {
	if pv.Size() != 4 { // a square has 4 sides
		return false
	}

	sz1 := sideLength(pv.At(0), pv.At(1))
	sz2 := sideLength(pv.At(1), pv.At(2))
	sz3 := sideLength(pv.At(2), pv.At(3))
	sz4 := sideLength(pv.At(3), pv.At(0))

	max := max(sz1, sz2, sz3, sz4)
	if sz1-sz2 > max/6 { // divide by 6 for an acceptable threshold
		return false
	}
	if sz1-sz3 > max/6 {
		return false
	}
	if sz1-sz4 > max/6 {
		return false
	}
	return true
}

// isBigger will compare two given boxes, and will return true
// if box1 if bigger than box2.
func isBigger(box1 gocv.PointVector, box2 gocv.PointVector) bool {
	m1 := 0
	m2 := 0
	for i := 1; i < box1.Size(); i++ {
		m1 += sideLength(box1.At(0), box1.At(i))
	}
	for i := 1; i < box2.Size(); i++ {
		m2 += sideLength(box2.At(0), box2.At(i))
	}
	return m1 > m2
}

// sortCoords will sort the coordinates so the order of the coordinates
// of the box is identical for each given box.
func sortCoords(box gocv.PointVector) gocv.PointVector {
	var tl, tr, br, bl image.Point
	for i := 0; i < box.Size(); i++ {
		r, l, b, t := 0, 0, 0, 0
		for j := 0; j < box.Size(); j++ {
			if j != i {
				if box.At(i).X <= box.At(j).X {
					r++
				} else {
					l++
				}
				if box.At(i).Y <= box.At(j).Y {
					t++
				} else {
					b++
				}
			}
		}
		if r > 1 && b > 1 {
			tl = box.At(i)
		}
		if l > 1 && b > 1 {
			tr = box.At(i)
		}
		if r > 1 && t > 1 {
			bl = box.At(i)
		}
		if l > 1 && t > 1 {
			br = box.At(i)
		}
	}
	tb := gocv.NewPointVector()
	tb.Append(bl)
	tb.Append(tl)
	tb.Append(tr)
	tb.Append(br)
	return tb
}

// biggestBoundingBox will return the biggest bounding box that
// is present in the given img matrix.
func biggestBoundingBox(img gocv.Mat) image.Rectangle {
	res := image.Rectangle{}
	cn := gocv.FindContours(img, gocv.RetrievalList, gocv.ChainApproxSimple)
	for i := 0; i < cn.Size(); i++ {
		r := gocv.BoundingRect(cn.At(i))
		if r.Size().X*r.Size().Y > res.Size().X*res.Size().Y {
			res = r
		}
	}
	return res
}

// addMargin adds given margin to the provided rectangle, making
// the rectangle area a bit bigger.
func addMargin(box image.Rectangle, margin int) image.Rectangle {
	if box.Min.X < margin {
		box.Min.X = margin
	}
	if box.Min.Y < margin {
		box.Min.Y = margin
	}
	return image.Rect(
		box.Min.X-margin,
		box.Min.Y-margin,
		box.Max.X+0,
		box.Max.Y+0,
	)
}
