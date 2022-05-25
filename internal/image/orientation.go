package image

import (
	"io"

	"github.com/rwcarlsen/goexif/exif"
	"gocv.io/x/gocv"
)

// fixOrientation will adjust the image according to the
// given orientation.
func fixOrientation(img gocv.Mat, o string) gocv.Mat {
	switch o {
	case "1":
		return img
	case "2":
		flip := gocv.NewMat()
		gocv.Flip(img, &flip, 0)
		img.Close()
		return flip
	case "3":
		rot := gocv.NewMat()
		gocv.Rotate(img, &rot, gocv.Rotate180Clockwise)
		img.Close()
		return rot
	case "4":
		flip := gocv.NewMat()
		gocv.Flip(img, &flip, 0)
		rot := gocv.NewMat()
		gocv.Rotate(flip, &rot, gocv.Rotate180Clockwise)
		img.Close()
		flip.Close()
		return rot
	case "5":
		flip := gocv.NewMat()
		gocv.Flip(img, &flip, 0)
		rot := gocv.NewMat()
		gocv.Rotate(flip, &rot, gocv.Rotate90Clockwise)
		img.Close()
		flip.Close()
		return rot
	case "6":
		rot := gocv.NewMat()
		gocv.Rotate(img, &rot, gocv.Rotate90Clockwise)
		img.Close()
		return rot
	case "7":
		flip := gocv.NewMat()
		gocv.Flip(img, &flip, 0)
		rot := gocv.NewMat()
		gocv.Rotate(flip, &rot, gocv.Rotate90CounterClockwise)
		img.Close()
		flip.Close()
		return rot
	case "8":
		rot := gocv.NewMat()
		gocv.Rotate(img, &rot, gocv.Rotate90CounterClockwise)
		img.Close()
		return rot
	}
	return img
}

// getOrientation tries to determine the orientation based on the exif
// info in case of a jpg image.
func getOrientation(r io.Reader) string {
	x, err := exif.Decode(r)
	if err != nil {
		return "1"
	}
	if x != nil {
		orient, err := x.Get(exif.Orientation)
		if err != nil {
			return "1"
		}
		if orient != nil {
			return orient.String()
		}
	}
	return "1"
}
