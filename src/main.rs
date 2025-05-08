#![feature(generic_const_exprs, core_intrinsics)]

use std::iter::zip;
use rand::distr::Uniform;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use tensoron::{tensor, ops::ML, Tensor, Matrix};

pub type R = f32;

pub trait Activation {
    fn activate(&self, m: Matrix<R>) -> Matrix<R>;
    fn derivative(&self, m: Matrix<R>) -> Matrix<R>;
}

#[derive(Debug, Clone)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn activate(&self, m: Matrix<R>) -> Matrix<R> {
        m.sigmoid()
    }
    fn derivative(&self, m: Matrix<R>) -> Matrix<R> {
        unimplemented!()
    }
}

pub struct Layer
{
    weights: Matrix<R>,
    biases: Matrix<R>,
    activation: Box<dyn Activation>,
    sz: usize,
}

impl std::fmt::Debug for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Layer(\n\tweights = {:#?},\n\tbiases = {:#?}\n)", self.weights, self.biases)
    }
}

// n = layer size, m = size of previous layer (size of input data for the first layer)
fn xavier_init(n: usize, m: usize) -> Tensor<R, 2> {
    let limit = 2.0f32.sqrt() / (n as f32);
    let mut rng = StdRng::from_os_rng();
    let uniform = Uniform::new(-limit, limit).unwrap();

    let values = (0..n * m).map(|_| rng.sample(&uniform)).collect();
    Tensor::from(([n, m], values))
}

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>
}

impl Network {
    pub fn new(sizes: Vec<usize>, activations: Vec<Box<dyn Activation>>) -> Self {
        let mut layers = vec![];

        for (s, a) in zip(sizes, activations) {
            layers.push(Layer {
                weights: tensor!([0, 0][]),
                biases: tensor!([0, 0][]),
                activation: a,
                sz: s,
            })
        }
        
        Self {
            layers
        }
    }

    pub fn forward(&mut self, x: Matrix<R>) -> Vec<Matrix<R>> {
        let mut acc = x;
        let mut things = vec![];

        for l in self.layers.iter() {
            let z = &(&l.weights * &acc) + &l.biases;
            let a = l.activation.activate(z);
            acc = a.clone();
            things.push(a);
        }

        let data = things.into_iter().map(|x| x.cpu()).collect();
        data
    }

    pub fn fit(&mut self, x: Matrix<R>) {
        let sizes: Vec<usize> = self.layers.iter().map(|l| l.sz).collect();

        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let sz_prev = if idx == 0 { x.shape()[0] } else { sizes[idx - 1] };
            let weights = xavier_init(layer.sz, sz_prev);
            let biases = xavier_init(layer.sz, 1);
            layer.weights = weights;
            layer.biases = biases;
        }
    }

    /*pub fn backprop(&mut self, outputs: Vec<Matrix<R>>, target: Matrix<R>) {
        let last = self.layers.last().unwrap();
        let mut deltas: Vec<Matrix<R>> = Vec::with_capacity(self.layers.len());

        let output = outputs.last().unwrap().clone();
        let output_error = (output - &target).scale(last.activation.derivative(output));
    }*/
}

fn main() {
    let xor_input = tensor!([4,2][
        0, 0,
        1, 0,
        0, 1,
        1, 1
    ]).map(|x| *x as f32);

    let sample = xor_input.view().slice([1]).to_tensor().transpose().cpu();
    let f = Box::new(Sigmoid);

    let mut net = Network::new(vec![4, 1], vec![f.clone(), f]);
    net.fit(sample.clone());

    println!("{:#?}", net);

    println!("{:#?}", net.forward(sample));
}
