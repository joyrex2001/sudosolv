package main

import (
	"fmt"

	"github.com/joyrex2001/sudosolv/internal/image"
	"github.com/joyrex2001/sudosolv/internal/numocr/dataset/generated"
	"github.com/joyrex2001/sudosolv/internal/numocr/network"
)

func main() {
	dataset := generated.NewGeneratedDataset()
	if err := network.Train(dataset); err != nil {
		fmt.Printf("error = %s\n", err)
		return
	}

	// dataset := mnist.NewMnistDataset()
	// if err := network.Train(dataset); err != nil {
	// 	fmt.Printf("error = %s\n", err)
	// 	return
	// }

	fontsets := []struct {
		name  string
		fonts []string
	}{
		{name: "mono", fonts: generated.FontsMono},
		{name: "sans", fonts: generated.FontsSans},
		{name: "serif", fonts: generated.FontsSerif},
		{name: "arial", fonts: generated.FontsArial},
		{name: "times", fonts: generated.FontsTimes},
		{name: "verdana", fonts: generated.FontsVerdana},
	}

	for _, f := range fontsets {
		fmt.Printf("training %s\n", f)
		d := generated.NewGeneratedDatasetForFont(f.name, f.fonts)
		if err := network.Train(d); err != nil {
			fmt.Printf("error while training %s = %s\n", f, err)
		}
	}

	for _, f := range fontsets {
		dataset := generated.NewGeneratedDatasetForFont(f.name, f.fonts)

		inf, err := network.NewInference(dataset)
		if err != nil {
			fmt.Printf("error = %s\n", err)
			return
		}

		img := image.NewPuzzleImage("_archive/IMG_6501.jpg")
		cel := img.GetSudokuCell(3, 2)
		// fmt.Printf("%v\n", cel)
		res, acc, err := inf.Predict(cel)
		if err != nil {
			fmt.Printf("error = %s\n", err)
			return
		}
		fmt.Printf("%3d (%f)\n", res, acc)
		// return

		for y := 0; y < 9; y++ {
			if y != 0 && y%3 == 0 {
				fmt.Printf("-----------+-----------+-----------\n")
			}
			for x := 0; x < 9; x++ {
				if x != 0 && x%3 == 0 {
					fmt.Printf("  |")
				}
				cel := img.GetSudokuCell(x, y)
				res, _, err := inf.Predict(cel)
				if err != nil {
					fmt.Printf("error = %s\n", err)
				}
				fmt.Printf("%3d", res)
			}
			fmt.Printf("\n")
		}
	}
}
