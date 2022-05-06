# sudosolv

Sudosolv will take a picture of a sudoku and will crop the sudoku from the image and decode this into plaintext.

To use this, you will need to train a number recognition classifier first. There are two datasets possible, either the mnist dataset, or a generated dataset based on available fonts.

Prepare the fonts dataset:
```shell
cd dataset/fonts
sh download.sh
sh create.sh
```

Prepare the mnist dataset:
```shell
cd dataset/mnist
sh download.sh
```

Train the network:
```shell
make train
```

Test with some sudoku image:
```shell
go run main.go decode -w trained.bin -f my_sudoku_image.jpg
go run main.go decode -w trained.bin -f my_sudoku_image --display
```