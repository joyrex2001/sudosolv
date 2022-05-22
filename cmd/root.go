package cmd

import (
	"fmt"
	"os"

	"github.com/joyrex2001/sudosolv/cmd/decode"
	"github.com/joyrex2001/sudosolv/cmd/server"
	"github.com/joyrex2001/sudosolv/cmd/train"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "sudosolv",
	Short: "sudosolv is a sudoku solver that takes a photo of a sudoku as input.",
}

func init() {
	rootCmd.AddCommand(train.Cmd)
	rootCmd.AddCommand(decode.Cmd)
	rootCmd.AddCommand(server.Cmd)
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
