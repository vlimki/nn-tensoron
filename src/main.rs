#![feature(generic_const_exprs, core_intrinsics)]

use rand::distr::Uniform;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::iter::zip;

use tensoron::ops::GpuAdd;
use tensoron::{ops::ML, tensor, Matrix, Tensor};

pub type R = f32;

pub trait Activation {
    fn activate(&self, m: &Matrix<R>) -> Matrix<R>;
    fn derivative(&self, m: &Matrix<R>) -> Matrix<R>;
}

#[derive(Debug, Clone)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn activate(&self, m: &Matrix<R>) -> Matrix<R> {
        m.sigmoid()
    }
    fn derivative(&self, m: &Matrix<R>) -> Matrix<R> {
        m.sigmoid_derivative()
    }
}

#[derive(Debug, Clone)]
pub struct ReLU;

impl Activation for ReLU {
    fn activate(&self, m: &Matrix<R>) -> Matrix<R> {
        m.relu()
    }
    fn derivative(&self, m: &Matrix<R>) -> Matrix<R> {
        m.relu_derivative()
    }
}

pub struct Layer {
    weights: Matrix<R>,
    biases: Matrix<R>,
    activation: Box<dyn Activation>,
    sz: usize,
}

impl std::fmt::Debug for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Layer(\n\tweights = {:#?},\n\tbiases = {:#?}\n)",
            self.weights, self.biases
        )
    }
}

// n = layer size, m = size of previous layer (size of input data for the first layer)
fn xavier_init(n: usize, m: usize) -> Tensor<R, 2> {
    let limit = 6.0f32.sqrt() / (n as f32 + m as f32).sqrt();
    let mut rng = StdRng::from_os_rng();
    let uniform = Uniform::new(-limit, limit).unwrap();

    let values = (0..n * m).map(|_| rng.sample(&uniform)).collect();
    Tensor::from(([n, m], values))
}

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
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

        Self { layers }
    }

    pub fn forward(&mut self, x: &Matrix<R>) -> (Vec<Matrix<R>>, Vec<Matrix<R>>) {
        let mut acc = x.clone();
        let mut zs = vec![acc.clone()];
        let mut a_vec = zs.clone();
        for layer in self.layers.iter() {
            let z = &(&layer.weights * &acc) + &layer.biases;
            let a = layer.activation.activate(&z);
            acc = a;
            zs.push(z);
            a_vec.push(acc.clone());
        }
        (zs, a_vec)
    }

    pub fn fit(&mut self, x: Matrix<R>) {
        let sizes: Vec<usize> = self.layers.iter().map(|l| l.sz).collect();

        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let sz_prev = if idx == 0 {
                x.shape()[0]
            } else {
                sizes[idx - 1]
            };
            let weights = xavier_init(layer.sz, sz_prev);
            let biases = xavier_init(layer.sz, 1);
            layer.weights = weights;
            layer.biases = biases;
        }
    }

    fn backprop(
        &mut self,
        (zs, outputs): (Vec<Matrix<R>>, Vec<Matrix<R>>),
        target: Matrix<R>,
    ) -> (Vec<Matrix<R>>, Vec<Matrix<R>>) {
        let mut delta = outputs.last().unwrap() - &target;
        delta = delta.gpu_cmul(
            &self
                .layers
                .last()
                .unwrap()
                .activation
                .derivative(zs.last().unwrap()),
        );

        let mut grad_w = Vec::new();
        let mut grad_b = Vec::new();

        for l in (0..self.layers.len()).rev() {
            let a_prev = &outputs[l];
            // weight gradient: delta · a_prev^T
            let dw = &delta * &a_prev.transpose();
            grad_w.push(dw);
            grad_b.push(delta.clone());
            if l > 0 {
                let w = &self.layers[l].weights;
                let prev = &w.transpose() * &delta;
                delta = prev.gpu_cmul(&self.layers[l].activation.derivative(&zs[l]));
            }
        }

        grad_w.reverse();
        grad_b.reverse();
        (grad_w, grad_b)
    }

    pub fn update(&mut self, grad_w: Vec<Matrix<R>>, grad_b: Vec<Matrix<R>>, lr: R) {
        for (idx, (dw, db)) in grad_w.iter().zip(grad_b).enumerate() {
            self.layers[idx].weights = &self.layers[idx].weights - &dw.scale(lr);
            self.layers[idx].biases = &self.layers[idx].biases - &db.scale(lr);
        }
    }

    pub fn predict(&mut self, input: &Matrix<R>) -> Matrix<R> {
        let outs = self.forward(input);
        let out = outs.1.last().unwrap().clone();
        out
    }

    pub fn train(&mut self, inputs: &[Matrix<R>], targets: &[Matrix<R>], epochs: usize, lr: R) {
        for _ in 0..epochs {
            let mut sum_dw: Vec<Matrix<R>> = self
                .layers
                .iter()
                .map(|l| l.weights.shape())
                .map(|s| Matrix::zeros(s))
                .collect();

            let mut sum_db: Vec<Matrix<R>> = self
                .layers
                .iter()
                .map(|l| l.biases.shape())
                .map(|s| Matrix::zeros(s))
                .collect();

            let n = inputs.len() as R;

            for (x, y) in inputs.iter().zip(targets.iter()) {
                let acts = self.forward(&x);
                let (dw, db) = self.backprop(acts, y.clone());
                for i in 0..sum_dw.len() {
                    sum_dw[i] = &sum_dw[i] + &dw[i];
                    sum_db[i] = &sum_db[i] + &db[i];
                }
            }

            for i in 0..self.layers.len() {
                sum_dw[i] = sum_dw[i].scale(1.0 / n);
                sum_db[i] = sum_db[i].scale(1.0 / n);
            }
            self.update(sum_dw, sum_db, lr);
        }
    }
}

fn data_to_colvecs(d: Matrix<R>) -> Vec<Matrix<R>> {
    let mut samples = vec![];
    for i in 0..d.shape()[0] {
        let sample = d.view().slice([i]).to_tensor().transpose();
        samples.push(sample);
    }

    samples
}

fn main() {
    let xor_input = tensor!([4,2][
        0, 0,
        1, 0,
        0, 1,
        1, 1
    ])
    .map(|x| *x as f32);

    let xor_output: Matrix<R> = tensor!([4,1][
        0.0,
        1.0,
        1.0,
        0.0
    ]);

    let sample_i = xor_input.view().slice([0]).to_tensor().transpose().cpu();

    let mut net = Network::new(vec![4, 1], vec![Box::new(Sigmoid), Box::new(Sigmoid)]);
    net.fit(sample_i.clone());

    let i = data_to_colvecs(xor_input.clone());
    let o = data_to_colvecs(xor_output);

    net.train(&i, &o, 5000, 1.0);

    for input in data_to_colvecs(xor_input.clone()) {
        let out = net.predict(&input).cpu();
        println!("Prediction: {:#?}", out);
    }
}
