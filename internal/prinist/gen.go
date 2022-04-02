package prinist

import (
	"fmt"
	"image"
	"image/color"
	"io/ioutil"

	"github.com/golang/freetype"
	"gorgonia.org/tensor"
)

var (
	width    = 28
	height   = 28
	fontsize = float64(37.0)
	fonts    = []string{
		"/Library/Fonts/Arial Unicode.ttf",
		"/System/Library/Fonts/NewYork.ttf",
		"/System/Library/Fonts/SFCompact.ttf",
		"/System/Library/Fonts/SFCompactRounded.ttf",
		"/System/Library/Fonts/SFNS.ttf",
		"/System/Library/Fonts/SFNSRounded.ttf",
		"/System/Library/Fonts/Symbol.ttf",
	}
)

// Gen will create an 28x28 byte buffer with the given number printed
// with the given ttf font.
func Gen(font string, number int) []byte {
	img := image.NewGray(image.Rect(0, 0, width, height))

	b, _ := ioutil.ReadFile(font)
	f, _ := freetype.ParseFont(b)

	ctx := freetype.NewContext()
	ctx.SetFont(f)
	ctx.SetFontSize(fontsize)
	ctx.SetClip(img.Bounds())
	ctx.SetDst(img)
	ctx.SetSrc(image.NewUniform(color.RGBA{255, 255, 255, 255}))
	pt := freetype.Pt(5, -10+int(ctx.PointToFixed(fontsize)>>6))
	ctx.DrawString(fmt.Sprintf("%d", number), pt)

	buf := []byte{}
	for _, x := range img.Pix {
		buf = append(buf, x)
	}

	// for i, x := range buf {
	// 	fmt.Printf(" %3d ", x)
	// 	if (i+1)%width == 0 {
	// 		fmt.Printf("\n")
	// 	}
	// }
	// fmt.Printf("\n")

	return buf
}

// GenXY will create both input and output data for various variations
// of fonts and numbers.
func GenXY() ([]byte, []float64) {
	x := []byte{}
	y := []float64{}
	y_ := map[int][]float64{}
	for i := 0; i <= 9; i++ {
		m := []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}
		m[i] = 0.9
		y_[i] = m
	}
	for _, f := range fonts {
		for i := 0; i <= 9; i++ {
			x = append(x, Gen(f, i)...)
			y = append(y, y_[i]...)
		}
	}
	return x, y
}

// X2Tensor converts a given []byte to a tensor with floats to use as
// input for the network.
func X2Tensor(M []byte) tensor.Tensor {
	cols := width * height
	rows := len(M) / cols
	x := make([]float64, len(M), len(M))
	for i, px := range M {
		max := 255.
		n := float64(px)/max*0.9 + 0.1
		if n == 1.0 {
			n = 0.999
		}
		x[i] = n
	}
	return tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(x))
}

// Y2Tensor converts a given []floats64 to a tensor to use as
// output for training.
func Y2Tensor(N []float64) tensor.Tensor {
	cols := 10
	rows := len(N) / cols
	return tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(N))
}
