package generated

import (
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"math/rand"
	"path"
	"regexp"
	"strings"

	"github.com/golang/freetype"
	"github.com/joyrex2001/sudosolv/internal/numocr/dataset"
	"golang.org/x/image/draw"
	"gorgonia.org/tensor"
)

const (
	width  = 28
	height = 28
)

// Fonts is a list of fonts that are available for training.
var Fonts = []string{
	"./internal/prinist/fonts/freefont-20100919/FreeMono.ttf",
	"./internal/prinist/fonts/freefont-20100919/FreeMonoBold.ttf",
	"./internal/prinist/fonts/freefont-20100919/FreeMonoBoldOblique.ttf",
	"./internal/prinist/fonts/freefont-20100919/FreeMonoOblique.ttf",
	"./internal/prinist/fonts/freefont-20100919/FreeSans.ttf",
	"./internal/prinist/fonts/freefont-20100919/FreeSansBold.ttf",
	"./internal/prinist/fonts/freefont-20100919/FreeSansBoldOblique.ttf",
	"./internal/prinist/fonts/freefont-20100919/FreeSansOblique.ttf",
	"./internal/prinist/fonts/freefont-20100919/FreeSerif.ttf",
	"./internal/prinist/fonts/freefont-20100919/FreeSerifBold.ttf",
	"./internal/prinist/fonts/freefont-20100919/FreeSerifBoldItalic.ttf",
	"./internal/prinist/fonts/freefont-20100919/FreeSerifItalic.ttf",

	"/System/Library/Fonts/Supplemental/Arial.ttf",
	"/System/Library/Fonts/Supplemental/Arial Bold.ttf",
	"/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
	"/System/Library/Fonts/Supplemental/Times New Roman.ttf",
	"/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
	"/System/Library/Fonts/Supplemental/Verdana.ttf",
}

// GeneratedDataset is the object that describes the generated
// fonts dataset.
type GeneratedDataset struct {
	size     int
	filename string
	basepath string
	epochs   int
	fonts    []string
}

// NewGeneratedDataset wil create a new GeneratedDataset instance.
func NewGeneratedDataset() dataset.Dataset {
	return &GeneratedDataset{
		size:     60000,
		epochs:   1,
		fonts:    Fonts,
		basepath: "./internal/numocr/dataset/generated/",
		filename: "trained.bin",
	}
}

// NewGeneratedDatasetForFont wil create a new GeneratedDataset
// instance for given font file.
func NewGeneratedDatasetForFont(font string) dataset.Dataset {
	re := regexp.MustCompile(`\..*$`)
	f := strings.ToLower(re.ReplaceAllString(path.Base(font), ""))
	re = regexp.MustCompile(`[^a-z0-9]`)
	f = re.ReplaceAllString(f, "")

	return &GeneratedDataset{
		size:     60000,
		epochs:   1,
		fonts:    []string{font},
		basepath: "./internal/numocr/dataset/generated/",
		filename: "trained" + f + ".bin",
	}
}

// WeightsFile will return the filename of the stored weights.
func (fd *GeneratedDataset) WeightsFile() string {
	return fd.basepath + fd.filename
}

// Epochs returns the number of epochs that should run during training.
func (fd *GeneratedDataset) Epochs() int {
	return fd.epochs
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
		x = append(x, gen(f, n)...)
		y = append(y, y_[n]...)
	}
	return x2tensor(x), y2tensor(y), nil
}

// gen will create an 28x28 byte buffer with the given number printed
// with the given ttf font.
func gen(font string, number int) []byte {
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
