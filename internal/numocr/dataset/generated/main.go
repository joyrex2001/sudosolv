package generated

import (
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"math/rand"

	"github.com/golang/freetype"
	"golang.org/x/image/draw"
	"gorgonia.org/tensor"

	"github.com/joyrex2001/sudosolv/internal/numocr/dataset"
)

const (
	width  = 28
	height = 28
)

// GeneratedDataset is the object that describes the generated
// fonts dataset.
type GeneratedDataset struct {
	size    int
	epochs  int
	noise   bool
	rndsize bool
	fonts   []string
}

// NewGeneratedDataset wil create a new GeneratedDataset instance.
func NewGeneratedDataset(fonts []string, epochs, size int, noise, rndsize bool) dataset.Dataset {
	return &GeneratedDataset{
		size:    size,
		epochs:  epochs,
		noise:   noise,
		rndsize: rndsize,
		fonts:   fonts,
	}
}

// Epochs returns the number of epochs that should run during training.
func (fd *GeneratedDataset) Epochs() int {
	return fd.epochs
}

// TestRatio returns how much can of the dataset can be used for
// validating the network.
func (fd *GeneratedDataset) TestRatio() float64 {
	return 0.1
}

// getFonts will return the configured fonts for this dataset.
func (fd *GeneratedDataset) getFonts() []string {
	return fd.fonts
}

// XY will create both input and output data for various variations
// of fonts and numbers.
func (fd *GeneratedDataset) XY() (tensor.Tensor, tensor.Tensor, error) {
	x := []byte{}
	y := []float64{}
	y_ := map[int][]float64{}
	for i := 0; i <= 9; i++ {
		m := []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}
		m[i] = 0.9
		y_[i] = m
	}
	fonts := fd.getFonts()
	for i := 0; i < fd.size; i++ {
		f := fonts[rand.Intn(len(fonts))]
		n := rand.Intn(10)
		b, err := fd.gen(f, n)
		if err != nil {
			return nil, nil, err
		}
		if fd.noise {
			b = noise(b, 512, 0.02) // 2% noise with intensity of 512
		}
		x = append(x, b...)
		y = append(y, y_[n]...)
	}
	return x2tensor(x), y2tensor(y), nil
}

// gen will create an 28x28 byte buffer with the given number printed
// with the given ttf font.
func (fd *GeneratedDataset) gen(font string, number int) ([]byte, error) {
	b, err := ioutil.ReadFile(font)
	if err != nil {
		return nil, err
	}
	f, err := freetype.ParseFont(b)
	if err != nil {
		return nil, err
	}

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

	s := 325.
	x, y := 50, 240

	if !fd.rndsize {
		s = float64(rand.Intn(50) + 200)
		x = rand.Intn(128)
		y = 256 - rand.Intn(100)
	}

	ctx.SetFontSize(s)
	ctx.DrawString(
		fmt.Sprintf("%d", number),
		freetype.Pt(x, y),
	)

	// resize to 28x28
	res := image.NewGray(image.Rect(0, 0, width, height))
	draw.NearestNeighbor.Scale(res, res.Rect, img, img.Bounds(), draw.Over, nil)
	buf := []byte{}
	for _, x := range res.Pix {
		buf = append(buf, x)
	}

	return buf, nil
}

// noise will add noise to the given buffer. The noise brightness level is
// indicated by amount and the chance is indicated the given float
// indicating a percentage (0...1).
func noise(buf []byte, amount int, chance float64) []byte {
	for i, v := range buf {
		if rand.Float64() > chance {
			continue
		}
		n := int(v) + (amount / 2) - rand.Intn(amount)
		if n < 0 {
			n = 0
		}
		if n > 255 {
			n = 255
		}
		buf[i] = byte(n)
	}
	return buf
}

// X2Tensor converts a given []byte to a tensor with floats to use as
// input for the network.
func x2tensor(M []byte) tensor.Tensor {
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
func y2tensor(N []float64) tensor.Tensor {
	cols := 10
	rows := len(N) / cols
	return tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(N))
}
