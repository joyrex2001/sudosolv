package server

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/joyrex2001/sudosolv/internal/server"
)

var Cmd = &cobra.Command{
	Use:   "server",
	Short: "HTTP server hosting the sudoku solver application.",
	Run:   serve,
}

func init() {
	Cmd.Flags().StringP("port", "p", ":8080", "Port to listen to")
	Cmd.Flags().StringP("weights", "w", "trained.bin", "Filename to write/update output weights")
}

func serve(cmd *cobra.Command, args []string) {
	port, _ := cmd.Flags().GetString("port")
	weights, _ := cmd.Flags().GetString("weights")
	if err := server.ListenAndServe(weights, port); err != nil {
		fmt.Printf("error running http server: %s\n", err)
		return
	}
}
