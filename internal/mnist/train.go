package mnist

import (
	"fmt"
	"log"
	"math/rand"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	"time"

	"gopkg.in/cheggaaa/pb.v1"
)

const (
	dataset = "train" // valid options are "train" or "test"
	bs      = 100     // batchsize
	dataloc = "./internal/mnist/dataset/"
)

type sli struct {
	start, end int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return 1 }

// Train will train a mnist network for given epochs and write the output
// to the given output filename.
func Train(output string, epochs int) error {
	rand.Seed(1337)

	var inputs, targets tensor.Tensor
	var err error

	if inputs, targets, err = loadMnist(dataset, dataloc, tensor.Float64); err != nil {
		return err
	}

	numExamples := inputs.Shape()[0]
	if err := inputs.Reshape(numExamples, 1, 28, 28); err != nil {
		return err
	}

	g := gorgonia.NewGraph()
	x := gorgonia.NewTensor(g, tensor.Float64, 4, gorgonia.WithShape(bs, 1, 28, 28), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(bs, 10), gorgonia.WithName("y"))

	m := newNetwork(g)
	if err := m.fwd(x); err != nil {
		return err
	}

	losses := gorgonia.Must(gorgonia.HadamardProd(m.out, y))
	cost := gorgonia.Must(gorgonia.Mean(losses))
	cost = gorgonia.Must(gorgonia.Neg(cost))

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

	for i := 0; i < epochs; i++ {
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
				return fmt.Errorf("Unable to slice x: %s", err)
			}

			if yVal, err = targets.Slice(sli{start, end}); err != nil {
				return fmt.Errorf("Unable to slice y: %s", err)
			}
			if err = xVal.(*tensor.Dense).Reshape(bs, 1, 28, 28); err != nil {
				return fmt.Errorf("Unable to reshape: %s", err)
			}

			gorgonia.Let(x, xVal)
			gorgonia.Let(y, yVal)
			if err := vm.RunAll(); err != nil {
				return fmt.Errorf("Failed at epoch %d: %s", i, err)
			}
			solver.Step(gorgonia.NodesToValueGrads(m.learnables()))
			vm.Reset()
			bar.Increment()
		}
		log.Printf("Epoch %d | cost %v", i, costVal)
	}

	return m.save(output)
}
