package server

import (
	"embed"
	"html/template"
	"log"
	"net/http"

	"github.com/joyrex2001/sudosolv/internal/image"
	"github.com/joyrex2001/sudosolv/internal/numocr/classifier"
	"github.com/joyrex2001/sudosolv/internal/sudoku"
)

//go:embed static/*
var assets embed.FS

type indexTemplate struct {
	Sudoku   string
	Solution string
	Error    string
}

// ListenAndServe will start the webserver and waits forever, unless
// an error occurs.
func ListenAndServe(weights, port string) error {
	inf, err := classifier.NewInference(weights)
	if err != nil {
		return err
	}

	mux := http.NewServeMux()
	mux.Handle("/static/", http.FileServer(http.FS(assets)))
	mux.HandleFunc("/", decode(inf))

	return http.ListenAndServe(port, mux)
}

// decode implements the /decode url
func decode(inf *classifier.Inference) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			renderTemplate(w, "static/index.tmpl", &indexTemplate{})
			return
		}

		if err := r.ParseMultipartForm(5 << 20); err != nil { // 5MB
			errorPage(w, err)
			return
		}

		data, _, err := r.FormFile("file")
		if err != nil {
			errorPage(w, err)
			return
		}
		defer data.Close()

		img, err := image.NewPuzzleImageFromReader(data)
		if err != nil {
			errorPage(w, err)
			return
		}

		sd, err := sudoku.NewSudokuFromPuzzleImage(img, inf)
		if err != nil {
			errorPage(w, err)
			return
		}

		msg := ""
		res := ""
		dec := sd.String()
		if sd.IsValid() {
			sd.Solve()
		} else {
			msg = "sudoku is invalid!\n\n"
		}
		res = sd.String()

		renderTemplate(w, "static/index.tmpl", &indexTemplate{Sudoku: dec, Solution: res, Error: msg})
	}
}

// errorPage will return a page containing the error.
func errorPage(w http.ResponseWriter, err error) {
	renderTemplate(w, "static/index.tmpl", &indexTemplate{Error: err.Error()})
}

// renderTemplate will output given template and renders it with the
// given values.
func renderTemplate(w http.ResponseWriter, name string, values interface{}) {
	t, err := template.ParseFS(assets, name)
	if err != nil {
		errorPage(w, err)
		return
	}
	w.WriteHeader(200)
	if err := t.Execute(w, values); err != nil {
		log.Printf("error rendering template: %s", err)
	}
}
