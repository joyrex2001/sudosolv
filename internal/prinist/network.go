package prinist

import (
	"encoding/gob"
	"fmt"
	"os"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// network is the internal representation of the cnn.
type network struct {
	g                  *gorgonia.ExprGraph
	w0, w1, w2, w3, w4 *gorgonia.Node // weights. the number at the back indicates which layer it's used for
	d0, d1, d2, d3     float64        // dropout probabilities
	out                *gorgonia.Node
	pred               gorgonia.Value
}

// newNetwork will return an instance of the convulational neural network.
func newNetwork(g *gorgonia.ExprGraph) *network {
	w0 := gorgonia.NewTensor(g, tensor.Float64, 4, gorgonia.WithShape(32, 1, 3, 3), gorgonia.WithName("w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w1 := gorgonia.NewTensor(g, tensor.Float64, 4, gorgonia.WithShape(64, 32, 3, 3), gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w2 := gorgonia.NewTensor(g, tensor.Float64, 4, gorgonia.WithShape(128, 64, 3, 3), gorgonia.WithName("w2"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w3 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(128*3*3, 625), gorgonia.WithName("w3"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w4 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(625, 10), gorgonia.WithName("w4"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	return &network{
		g:  g,
		w0: w0,
		w1: w1,
		w2: w2,
		w3: w3,
		w4: w4,
		d0: 0.2,
		d1: 0.2,
		d2: 0.2,
		d3: 0.55,
	}
}

// disableDropOut will disable the dropout nodes.
func (m *network) disableDropOut() {
	m.d0 = 0
	m.d1 = 0
	m.d2 = 0
	m.d3 = 0
}

// learnables will return the nodes that contain the tensors with the
// trained values.
func (m *network) learnables() gorgonia.Nodes {
	return gorgonia.Nodes{m.w0, m.w1, m.w2, m.w3, m.w4}
}

// save will save the current values of the learnables to a file
// with given filename.
func (m *network) save(fname string) error {
	f, err := os.Create(fname)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	for _, l := range m.learnables() {
		t := l.Value().(*tensor.Dense).Data()
		if err := enc.Encode(t); err != nil {
			return err
		}
	}
	return nil
}

// load will instantiate the learnables with the values that are
// stored in a file with given filename.
func (m *network) load(fname string) error {
	f, err := os.Open(fname)
	if err != nil {
		return err
	}
	defer f.Close()
	dec := gob.NewDecoder(f)
	for _, l := range m.learnables() {
		t := l.Value().(*tensor.Dense).Data().([]float64)
		var data []float64
		if err := dec.Decode(&data); err != nil {
			return err
		}
		if len(data) != len(t) {
			return fmt.Errorf("Unserialized length %d. Expected length %d", len(data), len(t))
		}
		copy(t, data)
	}
	return nil
}

// fwd will calculate the graph instantiated with given node.
func (m *network) fwd(x *gorgonia.Node) (err error) {
	var c0, c1, c2, fc *gorgonia.Node
	var a0, a1, a2, a3 *gorgonia.Node
	var p0, p1, p2 *gorgonia.Node
	var l0, l1, l2, l3 *gorgonia.Node

	// LAYER 0
	// here we convolve with stride = (1, 1) and padding = (1, 1),
	// which is your bog standard convolution for convnet
	if c0, err = gorgonia.Conv2d(x, m.w0, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return fmt.Errorf("Layer 0 Convolution failed: %s", err)
	}
	if a0, err = gorgonia.Rectify(c0); err != nil {
		return fmt.Errorf("Layer 0 activation failed: %s", err)
	}
	if p0, err = gorgonia.MaxPool2D(a0, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		return fmt.Errorf("Layer 0 Maxpooling failed: %s", err)
	}
	if l0, err = gorgonia.Dropout(p0, m.d0); err != nil {
		return fmt.Errorf("Unable to apply a dropout: %s", err)
	}

	// Layer 1
	if c1, err = gorgonia.Conv2d(l0, m.w1, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return fmt.Errorf("Layer 1 Convolution failed: %s", err)
	}
	if a1, err = gorgonia.Rectify(c1); err != nil {
		return fmt.Errorf("Layer 1 activation failed: %s", err)
	}
	if p1, err = gorgonia.MaxPool2D(a1, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		return fmt.Errorf("Layer 1 Maxpooling failed: %s", err)
	}
	if l1, err = gorgonia.Dropout(p1, m.d1); err != nil {
		return fmt.Errorf("Unable to apply a dropout to layer 1: %s", err)
	}

	// Layer 2
	if c2, err = gorgonia.Conv2d(l1, m.w2, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return fmt.Errorf("Layer 2 Convolution failed: %s", err)
	}
	if a2, err = gorgonia.Rectify(c2); err != nil {
		return fmt.Errorf("Layer 2 activation failed: %s", err)
	}
	if p2, err = gorgonia.MaxPool2D(a2, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		return fmt.Errorf("Layer 2 Maxpooling failed: %s", err)
	}

	var r2 *gorgonia.Node
	b, c, h, w := p2.Shape()[0], p2.Shape()[1], p2.Shape()[2], p2.Shape()[3]
	if r2, err = gorgonia.Reshape(p2, tensor.Shape{b, c * h * w}); err != nil {
		return fmt.Errorf("Unable to reshape layer 2: %s", err)
	}
	if l2, err = gorgonia.Dropout(r2, m.d2); err != nil {
		return fmt.Errorf("Unable to apply a dropout on layer 2: %s", err)
	}

	// Layer 3
	if fc, err = gorgonia.Mul(l2, m.w3); err != nil {
		return fmt.Errorf("Unable to multiply l2 and w3: %s", err)
	}
	if a3, err = gorgonia.Rectify(fc); err != nil {
		return fmt.Errorf("Unable to activate fc: %s", err)
	}
	if l3, err = gorgonia.Dropout(a3, m.d3); err != nil {
		return fmt.Errorf("Unable to apply a dropout on layer 3: %s", err)
	}

	// output decode
	var out *gorgonia.Node
	if out, err = gorgonia.Mul(l3, m.w4); err != nil {
		return fmt.Errorf("Unable to multiply l3 and w4: %s", err)
	}
	if m.out, err = gorgonia.SoftMax(out); err != nil {
		return fmt.Errorf("Unable to SoftMax: %s", err)
	}
	gorgonia.Read(m.out, &m.pred)

	return nil
}

// output will return the output of the graph after it has run.
func (m *network) output() ([]float64, error) {
	if m.pred == nil {
		return nil, fmt.Errorf("No output available")
	}
	return m.pred.Data().([]float64), nil
}
