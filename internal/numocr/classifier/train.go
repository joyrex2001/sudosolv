package classifier

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"gopkg.in/cheggaaa/pb.v1"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	"github.com/joyrex2001/sudosolv/internal/numocr/dataset"
)

const (
	bs = 100 // batchsize
)

type sli struct {
	start, end int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return 1 }

// Train will train a mnist network for given dataset object.
func Train(weights string, dataset dataset.Dataset) error {
	rand.Seed(1337)

	var err error

	epochs := dataset.Epochs()
	inputs, targets, err := dataset.XY()
	if err != nil {
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
	if err := m.fwd(x, true); err != nil {
		return err
	}

	if err := m.load(weights); err != nil {
		log.Printf("starting fresh, error loading previous snapshot: %s", err)
	}

	losses, err := gorgonia.HadamardProd(m.out, y)
	if err != nil {
		return err
	}
	cost := gorgonia.Must(gorgonia.Mean(losses))
	cost = gorgonia.Must(gorgonia.Neg(cost))

	var costVal gorgonia.Value
	gorgonia.Read(cost, &costVal)

	if _, err = gorgonia.Grad(cost, m.learnables()...); err != nil {
		return err
	}

	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.learnables()...))
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(bs)))
	defer vm.Close()

	batches := numExamples / bs
	testb := int(float64(batches) * dataset.TestRatio())
	trainb := batches - testb
	log.Printf("Batches %d/%d", trainb, testb)

	bar := pb.New(trainb)
	bar.SetRefreshRate(time.Second)
	bar.SetMaxWidth(80)

	for i := 0; i < epochs; i++ {
		bar.Prefix(fmt.Sprintf("Epoch %d", i))
		bar.Set(0)
		bar.Start()
		for b := 0; b < trainb; b++ {
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
			if err = xVal.(*tensor.Dense).Reshape(bs, 1, 28, 28); err != nil {
				return fmt.Errorf("Unable to reshape: %s", err)
			}

			if yVal, err = targets.Slice(sli{start, end}); err != nil {
				return fmt.Errorf("Unable to slice y: %s", err)
			}

			gorgonia.Let(x, xVal)
			gorgonia.Let(y, yVal)
			if err := vm.RunAll(); err != nil {
				return fmt.Errorf("Failed at epoch %d: %s", i, err)
			}
			if err := solver.Step(gorgonia.NodesToValueGrads(m.learnables())); err != nil {
				return fmt.Errorf("Unable to solve: %s", err)
			}
			vm.Reset()
			bar.Increment()
		}
		log.Printf("Epoch %d | cost %v", i, costVal)

		// save newly learned weights
		if err := m.save(weights); err != nil {
			return err
		}

		// test network performance
		xt, err := inputs.Slice(sli{trainb * bs, numExamples})
		if err != nil {
			return err
		}
		yt, err := targets.Slice(sli{trainb * bs, numExamples})
		if err != nil {
			return err
		}
		pred, vals, score, err := testNetwork(weights, dataset, xt, yt)
		if err != nil {
			return err
		}
		log.Println()
		log.Printf("pred = %v\nvals = %v\nscore = %f\n\n", pred, vals, score)
	}
	bar.Finish()

	return nil
}

func testNetwork(weights string, dataset dataset.Dataset, x tensor.Tensor, y tensor.Tensor) ([]int, []int, float64, error) {
	g := gorgonia.NewGraph()
	in := gorgonia.NewTensor(g, tensor.Float64, 4, gorgonia.WithShape(1, 1, 28, 28), gorgonia.WithName("x"))
	nn := newNetwork(g)
	if err := nn.fwd(in, false); err != nil {
		return nil, nil, 0, err
	}

	if err := nn.load(weights); err != nil {
		return nil, nil, 0, err
	}
	vm := gorgonia.NewTapeMachine(g)

	pred := make([]int, 10)
	vals := make([]int, 10)

	ok := 0
	total := x.Shape()[0]
	for i := 0; i < total; i++ {
		vm.Reset()
		x_, err := x.Slice(sli{i, i + 1})
		if err != nil {
			return nil, nil, 0, err
		}
		if err := x_.(*tensor.Dense).Reshape(1, 1, 28, 28); err != nil {
			return nil, nil, 0, err
		}
		if err := gorgonia.Let(in, x_); err != nil {
			return nil, nil, 0, err
		}
		if err := vm.RunAll(); err != nil {
			return nil, nil, 0, err
		}
		res, err := nn.output()
		if err != nil {
			return nil, nil, 0, err
		}
		y_, err := y.Slice(sli{i, i + 1})
		if err != nil {
			return nil, nil, 0, err
		}
		m := bestMatch(res)
		v := bestMatch(y_.Data().([]float64))
		if m == v {
			ok++
		}
		pred[m]++
		vals[v]++
	}
	score := float64(ok) / float64(total)
	return pred, vals, score, nil
}

func bestMatch(v []float64) int {
	res := 0
	ms := float64(0.)
	for n, s := range v {
		if s > ms {
			res = n
			ms = s
		}
	}
	return res
}
