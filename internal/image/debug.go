package image

import (
	"fmt"
	"runtime"

	"gocv.io/x/gocv"
)

func init() {
	runtime.LockOSThread()
}

// display will display the given OpenCV matrix in a new window
// and will wait until a keypres.
func display(img gocv.Mat) {
	for {
		window := gocv.NewWindow("Hello")
		window.IMShow(img)
		if window.WaitKey(10000) >= 0 {
			return
		}
	}
}

// logPointVector will print the coordinates in the given PointVector.
func logPointVector(pv gocv.PointVector) {
	fmt.Printf("PointVector:\n")
	for i := 0; i < pv.Size(); i++ {
		fmt.Printf("  %#v\n", pv.At(i))
	}
	fmt.Printf("\n")
}
