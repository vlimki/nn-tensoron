use std::iter::zip;

use tensoron::{tensor, ops::ML, Tensor, Matrix};

pub type R = f32;

pub trait Activation {
    fn activate(&self, m: Matrix<R>) -> Matrix<R>;
    fn derivative(&self, m: Matrix<R>) -> Matrix<R>;
}

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

impl Layer {
}

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
}

fn main() {}
