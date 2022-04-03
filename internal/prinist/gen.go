package prinist

import (
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"math/rand"

	"github.com/golang/freetype"
	"golang.org/x/image/draw"
	"gorgonia.org/tensor"
)

const (
	width  = 28
	height = 28
)

var (
	fonts = []string{
		"/Library/Fonts/Arial Unicode.ttf",
		"/System/Library/Fonts/NewYork.ttf",
		"/System/Library/Fonts/SFCompact.ttf",
		"/System/Library/Fonts/SFCompactRounded.ttf",
		"/System/Library/Fonts/SFNS.ttf",
		"/System/Library/Fonts/SFNSRounded.ttf",
	}
)

// Gen will create an 28x28 byte buffer with the given number printed
// with the given ttf font.
func Gen(font string, number int) []byte {
	b, _ := ioutil.ReadFile(font)
	f, _ := freetype.ParseFont(b)
	ctx := freetype.NewContext()
	img := image.NewGray(image.Rect(0, 0, 256, 256))
	ctx.SetFont(f)
	ctx.SetClip(img.Bounds())
	ctx.SetDst(img)

	// print number at random pos with random size and color
	ctx.SetSrc(image.NewUniform(color.RGBA{
		uint8(rand.Intn(50) + 205),
		uint8(rand.Intn(50) + 205),
		uint8(rand.Intn(50) + 205),
		uint8(rand.Intn(50) + 205),
	}))
	ctx.SetFontSize(float64(rand.Intn(50) + 200))
	ctx.DrawString(
		fmt.Sprintf("%d", number),
		freetype.Pt(rand.Intn(128), 256-rand.Intn(100)),
	)

	// resize to 28x28
	res := image.NewGray(image.Rect(0, 0, width, height))
	draw.NearestNeighbor.Scale(res, res.Rect, img, img.Bounds(), draw.Over, nil)
	buf := []byte{}
	for _, x := range res.Pix {
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
func GenXY(size int) ([]byte, []float64) {
	x := []byte{}
	y := []float64{}
	y_ := map[int][]float64{}
	for i := 0; i <= 9; i++ {
		m := []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}
		m[i] = 0.9
		y_[i] = m
	}
	for i := 0; i < size; i++ {
		f := fonts[rand.Intn(len(fonts))]
		n := rand.Intn(10)
		x = append(x, Gen(f, n)...)
		y = append(y, y_[n]...)
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
