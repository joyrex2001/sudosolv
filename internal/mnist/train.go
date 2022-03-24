package mnist

import (
	"flag"
	"fmt"
	"log"
	"math/rand"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	"time"

	"gopkg.in/cheggaaa/pb.v1"
)

var (
	epochs    = flag.Int("epochs", 4, "Number of epochs to train for")
	dataset   = flag.String("dataset", "train", "Which dataset to train on? Valid options are \"train\" or \"test\"")
	batchsize = flag.Int("batchsize", 100, "Batch size")
)

const loc = "./internal/mnist/dataset/"
const backup = "/tmp/mnist-trained.bin"

type sli struct {
	start, end int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return 1 }

func Train() {
	flag.Parse()
	rand.Seed(1337)

	var inputs, targets tensor.Tensor
	var err error

	trainOn := *dataset
	if inputs, targets, err = Load(trainOn, loc, tensor.Float64); err != nil {
		log.Fatal(err)
	}

	// the data is in (numExamples, 784).
	// In order to use a convnet, we need to massage the data
	// into this format (batchsize, numberOfChannels, height, width).
	//
	// This translates into (numExamples, 1, 28, 28).
	//
	// This is because the convolution operators actually understand height and width.
	//
	// The 1 indicates that there is only one channel (MNIST data is black and white).
	numExamples := inputs.Shape()[0]
	bs := *batchsize
	// todo - check bs not 0

	if err := inputs.Reshape(numExamples, 1, 28, 28); err != nil {
		log.Fatal(err)
	}
	g := gorgonia.NewGraph()
	x := gorgonia.NewTensor(g, tensor.Float64, 4, gorgonia.WithShape(bs, 1, 28, 28), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(bs, 10), gorgonia.WithName("y"))
	m := newConvNet(g)
	if err = m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}
	losses := gorgonia.Must(gorgonia.HadamardProd(m.out, y))
	cost := gorgonia.Must(gorgonia.Mean(losses))
	cost = gorgonia.Must(gorgonia.Neg(cost))

	// we wanna track costs
	var costVal gorgonia.Value
	gorgonia.Read(cost, &costVal)

	prog, locMap, _ := gorgonia.Compile(g)
	log.Printf("%v", prog)

	vm := gorgonia.NewTapeMachine(g, gorgonia.WithPrecompiled(prog, locMap), gorgonia.BindDualValues(m.learnables()...))
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(bs)))
	defer vm.Close()

	batches := numExamples / bs
	log.Printf("Batches %d", batches)
	bar := pb.New(batches)
	bar.SetRefreshRate(time.Second)
	bar.SetMaxWidth(80)

	for i := 0; i < *epochs; i++ {
		bar.Prefix(fmt.Sprintf("Epoch %d", i))
		bar.Set(0)
		bar.Start()
		for b := 0; b < batches; b++ {
			start := b * bs
			end := start + bs
			if start >= numExamples {
				break
			}
			if end > numExamples {
				end = numExamples
			}

			var xVal, yVal tensor.Tensor
			if xVal, err = inputs.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice x")
			}

			if yVal, err = targets.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice y")
			}
			if err = xVal.(*tensor.Dense).Reshape(bs, 1, 28, 28); err != nil {
				log.Fatalf("Unable to reshape %v", err)
			}

			gorgonia.Let(x, xVal)
			gorgonia.Let(y, yVal)
			if err = vm.RunAll(); err != nil {
				log.Fatalf("Failed at epoch  %d: %v", i, err)
			}
			solver.Step(gorgonia.NodesToValueGrads(m.learnables()))
			vm.Reset()
			bar.Increment()
		}
		log.Printf("Epoch %d | cost %v", i, costVal)
	}
	if err := m.save(backup); err != nil {
		log.Fatal(err)
	}
}
