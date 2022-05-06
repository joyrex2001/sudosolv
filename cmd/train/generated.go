package train

import (
	"fmt"
	"io/ioutil"
	"log"
	"path"
	"strings"

	"github.com/joyrex2001/sudosolv/internal/numocr/dataset/generated"
	"github.com/joyrex2001/sudosolv/internal/numocr/network"

	"github.com/spf13/cobra"
)

var generatedCmd = &cobra.Command{
	Use:   "generated",
	Short: "Train a network for number recognition using a generated dataset.",
	Run:   trainGenerated,
}

func init() {
	Cmd.AddCommand(generatedCmd)
	generatedCmd.Flags().Int("size", 60000, "Size of the dataset to be generated")
	generatedCmd.Flags().Bool("noise", false, "Add noise to the dataset")
	generatedCmd.Flags().Bool("rndsize", false, "Randomly vary the size of the fonts")
	generatedCmd.Flags().String("dataloc", "", "Path where to be used ttf font files are located")
	Cmd.MarkFlagRequired("fonts")
}

func trainGenerated(cmd *cobra.Command, args []string) {
	weights, _ := cmd.Flags().GetString("weights")
	epochs, _ := cmd.Flags().GetInt("epochs")

	size, _ := cmd.Flags().GetInt("size")
	noise, _ := cmd.Flags().GetBool("noise")
	rndsize, _ := cmd.Flags().GetBool("rndsize")
	fpath, _ := cmd.Flags().GetString("dataloc")
	fonts, err := getFontsFromFolder(fpath)
	if err != nil {
		log.Fatal(err)
	}

	dataset := generated.NewGeneratedDataset(fonts, epochs, size, noise, rndsize)
	if err := network.Train(weights, dataset); err != nil {
		fmt.Printf("error = %s\n", err)
		return
	}
}

func getFontsFromFolder(folder string) ([]string, error) {
	fonts := []string{}

	files, err := ioutil.ReadDir(folder)
	if err != nil {
		return nil, err
	}

	for _, file := range files {
		if strings.ToLower(path.Ext(file.Name())) == ".ttf" {
			fonts = append(fonts, path.Join(folder, file.Name()))
		}
	}

	if len(fonts) == 0 {
		return nil, fmt.Errorf("no ttf files found in %s", folder)
	}
	return fonts, nil
}
