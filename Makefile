run:
	go run main.go 

train:
	go run main.go train generated -w trained.bin --size 30000 --epochs 5 --fonts ./dataset/fonts/freefont-20100919
	go run main.go train generated -w trained.bin --size 30000 --epochs 5 --fonts ./dataset/fonts/freefont-20100919 --rndsize
	go run main.go train generated -w trained.bin --size 30000 --epochs 5 --fonts ./dataset/fonts/freefont-20100919 --noise
	go run main.go train generated -w trained.bin --size 30000 --epochs 5 --fonts ./dataset/fonts/freefont-20100919
	go run main.go train generated -w trained.bin --size 30000 --epochs 5 --fonts ./dataset/fonts/freefont-20100919 --rndsize --noise
	go run main.go train generated -w trained.bin --size 30000 --epochs 5 --fonts ./dataset/fonts/freefont-20100919
	go run main.go train generated -w trained.bin --size 30000 --epochs 5 --fonts ./dataset/fonts/freefont-20100919 --rndsize
	go run main.go train generated -w trained.bin --size 30000 --epochs 5 --fonts ./dataset/fonts/freefont-20100919 --noise	

build:
	go build -o sudosolv

clean:
	rm -f sudosolv
	rm -rf dist
	go mod tidy
	rm -f coverage.out
	go clean -testcache

cloc:
	cloc --exclude-dir=vendor,node_modules,dist,_notes,_archive .

fmt:
	find ./internal -type f -name \*.go -exec gofmt -s -w {} \;
	go fmt ./...

test:
	go vet ./...
	go test ./... -cover

lint:
	golint ./internal/...
	# errcheck ./internal/... ./cmd/...

cover:
	go test ./... -cover -coverprofile=coverage.out
	go tool cover -html=coverage.out
	
deps:
	go install golang.org/x/lint/golint@latest
	go install github.com/kisielk/errcheck@latest
	go install github.com/mitchellh/gox@latest
	go install github.com/tcnksm/ghr@latest

.PHONY: run build train clean cloc fmt test lint cover deps