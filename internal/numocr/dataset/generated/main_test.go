package generated

import (
	"fmt"
	"testing"
)

func TestFontCompatability(t *testing.T) {
	fonts := allFonts()
	// fonts := FontsSuplemental
	for i, f := range fonts {
		fmt.Printf("[%03d] testing %s\n", i, f)
		c := 0
		for n := 0; n < 10; n++ {
			buf, err := gen(f, n)
			if err != nil {
				t.Errorf("error generating %d with %s: %s", n, f, err)
			}
			x := 0
			for j, v := range buf {
				x += j * int(v)
			}
			if x == 0 {
				t.Errorf("font %s does not contain number %d", f, n)
			}
			if x != 0 && x == c {
				t.Errorf("font %s seems to have duplicate %d", f, n)
			}
			c = x
			// printbuf(buf)
		}
	}
}

func TestNoise(t *testing.T) {
	fonts := FontsArial
	for _, f := range fonts {
		for n := 0; n < 10; n++ {
			buf, err := gen(f, n)
			if err != nil {
				t.Errorf("error generating %d with %s: %s", n, f, err)
			}
			buf = noise(buf, 255, 0.05)
			// printbuf(buf)
		}
	}
}

func printbuf(buf []byte) {
	for j, x := range buf {
		fmt.Printf(" %3d ", x)
		if (j+1)%width == 0 {
			fmt.Printf("\n")
		}
	}
	fmt.Printf("\n")
}
