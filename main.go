package main

import (
	"fmt"
	"image"

	"gocv.io/x/gocv"
)

func display(img gocv.Mat) {
	for {
		window := gocv.NewWindow("Hello")
		window.IMShow(img)
		if window.WaitKey(10000) >= 0 {
			return
		}
	}
}

func logPointVector(pv gocv.PointVector) {
	fmt.Printf("PointVector:\n")
	for i := 0; i < pv.Size(); i++ {
		fmt.Printf("  %#v\n", pv.At(i))
	}
	fmt.Printf("\n")
}

func max(x ...int) int {
	max := 0
	for _, n := range x {
		if n > max {
			max = n
		}
	}
	return max
}

func sideLength(a, b image.Point) int {
	return (a.X-b.X)*(a.X-b.X) + (a.Y-b.Y)*(a.Y-b.Y)
}

func isSquare(pv gocv.PointVector) bool {
	if pv.Size() != 4 {
		return false
	}

	sz1 := sideLength(pv.At(0), pv.At(1))
	sz2 := sideLength(pv.At(1), pv.At(2))
	sz3 := sideLength(pv.At(2), pv.At(3))
	sz4 := sideLength(pv.At(3), pv.At(0))

	max := max(sz1, sz2, sz3, sz4)
	if sz1-sz2 > max/6 {
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

func main() {
	img := gocv.IMRead("_archive/IMG_6502.jpg", gocv.IMReadColor)
	defer img.Close()

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
		apx := gocv.ApproxPolyDP(cn.At(i), 450, true)
		if isSquare(apx) && isBigger(apx, box) {
			box = sortCoords(apx)
		}
	}

	// bx := gocv.NewPointsVector()
	// bx.Append(box)
	// logPointVector(box)
	// gocv.DrawContours(&img, bx, -1, color.RGBA{0, 255, 0, 0}, 10)
	// display(img)

	tb := gocv.NewPointVector()
	tb.Append(image.Point{0, 0})
	tb.Append(image.Point{0, 500})
	tb.Append(image.Point{500, 500})
	tb.Append(image.Point{500, 0})
	pt := gocv.GetPerspectiveTransform(box, tb)

	crop := gocv.NewMat()
	gocv.WarpPerspective(img, &crop, pt, image.Point{500, 500})

	display(crop)
}
