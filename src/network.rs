use std::time::Instant;

use ndarray::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::{ParallelIterator, IntoParallelRefMutIterator};

const SEED: u64 = 42;

pub struct AdamOptimizer {
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64
}

pub enum Optimizer {
    SGD,
    Adam(AdamOptimizer)
}

pub enum Regularization {
    None,
    L1(f64),
    L2(f64)
}

pub struct Network {
    pub sizes: Vec<usize>,
    pub weights: Vec<Array2<f64>>,
    pub biases: Vec<Array2<f64>>,
    pub optimizer: Optimizer,
    w_epsilon1: Vec<Array2<f64>>,
    w_epsilon2: Vec<Array2<f64>>,
    b_epsilon1: Vec<Array2<f64>>,
    b_epsilon2: Vec<Array2<f64>>
}

fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

fn softmax(x: &ArrayView2<f64>) -> Array2<f64> {
    let expx = x.mapv(f64::exp);
    &expx / expx.sum()
}

fn softmax_dLdZ(output: &ArrayView2<f64>, target: &ArrayView2<f64>) -> Array2<f64> {
    output - target
}

fn cross_entropy(y_true: &ArrayView1<f64>, y_pred: &ArrayView1<f64>) -> f64 {
    let y_pred = y_pred.mapv(|x| (x.clamp(1e-12, 1.0) + 1e-9).ln());
    let n = y_true.len();
    -(y_true * y_pred).sum() / n as f64
}

pub fn cost(output: &ArrayView1<f64>, target: &ArrayView1<f64>) -> f64 {
    cross_entropy(target, output)
}

fn sigmoid(x: &ArrayView2<f64>) -> Array2<f64> {
    1.0 / (1.0 + x.mapv(|x| (-x).exp()))
}

fn sigmoid_prime(x: &ArrayView2<f64>) -> Array2<f64> {
    let s = sigmoid(x);
    &s * &(1.0 - &s)
}

fn max_index(a: &ArrayView1<f64>) -> usize {
    let mut max = a[0];
    let mut max_index = 0;
    let n = a.shape()[0];
    for i in 1..n {
        if a[i] > max {
            max = a[i];
            max_index = i;
        }
    }
    max_index
}

impl Network {
    pub fn new(sizes: Vec<usize>, optimizer: Optimizer) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
        let n_layers = sizes.len();
        let mut weights = Vec::with_capacity(n_layers - 1);
        let mut biases = Vec::with_capacity(n_layers - 1);
        for i in 1..sizes.len() {
            let w = 
                Array::from_shape_simple_fn((sizes[i], sizes[i - 1]), || {
                    (2.0 / sizes[i - 1] as f64).sqrt() * rng.sample::<f64, _>(StandardNormal)
                });
            let b = Array::zeros((sizes[i], 1));
            weights.push(w);
            biases.push(b);
        }
        let mut w_epsilon1 = Vec::new();
        let mut w_epsilon2 = Vec::new();
        let mut b_epsilon1 = Vec::new();
        let mut b_epsilon2 = Vec::new();
        if let Optimizer::Adam(_) = optimizer {
            w_epsilon1.reserve(n_layers);
            w_epsilon2.reserve(n_layers);
            b_epsilon1.reserve(n_layers);
            b_epsilon2.reserve(n_layers);
            for i in 1..n_layers {
                w_epsilon1.push(Array::zeros((sizes[i], sizes[i - 1])));
                w_epsilon2.push(Array::zeros((sizes[i], sizes[i - 1])));
                b_epsilon1.push(Array::zeros((sizes[i], 1)));
                b_epsilon2.push(Array::zeros((sizes[i], 1)));
            }
        }
        Self {
            sizes,
            weights,
            biases,
            optimizer,
            w_epsilon1,
            w_epsilon2,
            b_epsilon1,
            b_epsilon2
        }
    }

    // fn execute_minibatch(&self, input: &ArrayView2<f64>, target: &ArrayView2<f64>, b: usize) -> (f64, Vec<Array2<f64>>, Vec<Array2<f64>>) {
    //     let example = input.slice(s![b, ..]);
    //     let example_class = target.slice(s![b, ..]);

    //     let (output, z_s, a_s) = 
    //         self.forward_pass(&example);

    //     let (gw, gb) = 
    //         self.backward_pass(&output.view(), &example_class, &z_s, &a_s);
        
    //     (cost(&output.view(), &example_class), gw, gb)
    // }

    pub fn train(&mut self, training_data: &ArrayView2<f64>, 
        training_class: &ArrayView2<f64>, val_data: &ArrayView2<f64>, 
        val_class: &ArrayView2<f64>, epochs: usize, 
        mini_batch_size: usize, eta: f64, decay_rate: f64,
        regularization: &Regularization
    ) {
        let mut iteration_index = 0;
        let mut eta_current = eta;

        let n_layers = self.sizes.len();
        let n = training_data.shape()[0];
        for j in 0..epochs {
            let start = Instant::now();
            println!("Epoch {}", j);
            let mut loss_avg = 0.0;

            let num_mini_batches = ceil_div(n, mini_batch_size);
            let mut mini_batches = Vec::with_capacity(num_mini_batches);
            for i in 0..num_mini_batches {
                let start = i * mini_batch_size;
                let mut end = (i + 1) * mini_batch_size;
                if end > n {
                    end = n;
                }
                let mini_batch_data = training_data.slice(s![start..end, ..]);
                let mini_batch_class = training_class.slice(s![start..end, ..]);
                mini_batches.push((mini_batch_data, mini_batch_class));
            }
            // println!("Processed minibatches in {}s", start.elapsed().as_secs_f32());
            for (input, target) in mini_batches {
                let m = input.shape()[0];
                let mut gw_sum = Vec::with_capacity(n_layers - 1);
                let mut gb_sum = Vec::with_capacity(n_layers - 1);
                for i in 0..n_layers - 1 {
                    gw_sum.push(Array::zeros(self.weights[i].dim()));
                    gb_sum.push(Array::zeros(self.biases[i].dim()));
                }
                
                // Serial
                for b in 0..m {
                    let example = input.slice(s![b, ..]);
                    let example_class = target.slice(s![b, ..]);

                    let (output, z_s, a_s) = 
                        self.forward_pass(&example);

                    let (gw, gb) = 
                        self.backward_pass(&output.view(), &example_class, &z_s, &a_s);
                    
                    loss_avg += cost(&output.view(), &example_class);
                    for i in 0..n_layers - 1 {
                        gw_sum[i] += &gw[i];
                        gb_sum[i] += &gb[i];
                    }
                }

                // Parallel
                // let outputs = 
                //     (0..m).into_par_iter().map(|b| self.execute_minibatch(&input, &target, b)).reduce(
                //         || (0.0, gw_sum.clone(), gb_sum.clone()), 
                //         |mut a, b| {
                //         for i in 0..n_layers - 1 {
                //             a.1[i] = &a.1[i] + &b.1[i];
                //             a.2[i] = &a.2[i] + &b.2[i];
                //         }
                //         a.0 = a.0 + b.0;
                //         a
                //     });

                // loss_avg += outputs.0;
                // let mut gw_sum = outputs.1;
                // let mut gb_sum = outputs.2;

                gw_sum.par_iter_mut().for_each(|x| *x /= m as f64);
                gb_sum.par_iter_mut().for_each(|x| *x /= m as f64);

                if decay_rate > 0.0 {
                    eta_current = eta * f64::exp(-decay_rate * iteration_index as f64);
                }
                iteration_index += 1;

                self.update_network(iteration_index, &gw_sum, &gb_sum, n, eta_current, &regularization)
            }

            println!("Epoch {} complete in {}s", j, start.elapsed().as_secs_f32());
            println!("Loss: {}", loss_avg / n as f64);
            if j % 10 == 0 {
                self.eval_network(&val_data, &val_class);
            }
        }
    }

    pub fn eval_network(&self, validation_data: &ArrayView2<f64>, validation_class: &ArrayView2<f64>) {
        let n = validation_data.shape()[0];
        let mut loss_avg = 0.0;
        let mut tp = 0.0;
        for i in 0..n {
            let example = 
                validation_data.slice(s![i, ..]);
            let example_class = 
                validation_class.slice(s![i, ..]);
            let example_class_num = max_index(&example_class);
            let (output, _, _) = 
                self.forward_pass(&example);
            let output_num = max_index(&output.view());
            if output_num == example_class_num {
                tp += 1.0;
            }
            loss_avg += cost(&output.view(), &example_class);
        }
        println!("Validation Loss: {}", loss_avg / n as f64);
        println!("Classification Accuracy: {}", tp / n as f64);
    }

    pub fn predict(&self, input: &ArrayView1<f64>) -> Array1<f64> {
        let (output, _, _) = self.forward_pass(input);
        output
    }

    pub fn forward_pass(&self, input: &ArrayView1<f64>) -> (Array1<f64>, Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let n_layers = self.sizes.len();
        let mut z_s: Vec<Array2<f64>> = Vec::with_capacity(n_layers);
        let mut a_s: Vec<Array2<f64>> = Vec::with_capacity(n_layers);
        let input = 
            input.insert_axis(Axis(1));
        z_s.push(input.to_owned());
        a_s.push(input.to_owned());
        let n = n_layers - 2;
        assert!(n <= self.weights.len());
        assert!(n <= self.biases.len());
        for i in 0..n {
            let a = &a_s[i].view();
            z_s.push(self.weights[i].dot(a) + &self.biases[i]);
            let z = &z_s[i + 1].view();
            a_s.push(sigmoid(&z));
        }
        assert!(n + 1 < a_s.len());
        assert!(n + 1 < z_s.len());
        let a = &a_s[n].view();
        z_s.push(self.weights[n].dot(a) + &self.biases[n]);
        let z = &z_s[n + 1].view();
        a_s.push(softmax(&z));
        let a = &a_s[n + 1].view();
        let outputs = a.slice(s![.., 0]).to_owned();
        (outputs, z_s, a_s)
    }

    pub fn backward_pass(&self, outputs: &ArrayView1<f64>, targets: &ArrayView1<f64>, 
        zs: &Vec<Array2<f64>>, activations: &Vec<Array2<f64>>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
            let n_layers = self.sizes.len();
            let mut gw = vec![Array2::zeros((0, 0)); n_layers - 1];
            let mut gb = vec![Array2::zeros((0, 0)); n_layers - 1];
            let outputs = outputs.insert_axis(Axis(1));
            let targets = targets.insert_axis(Axis(1));
            let mut delta = softmax_dLdZ(&outputs, &targets);
            let n = n_layers - 2;
            assert!(self.weights.len() == n + 1);
            assert!(gw.len() == n + 1);
            assert!(gb.len() == n + 1);
            gw[n] = delta.dot(&activations[n].t());
            gb[n] = delta.to_owned();
            for l in (1..=n).rev() {
                delta = self.weights[l].t().dot(&delta) * sigmoid_prime(&zs[l].view());
                gw[l - 1] = delta.dot(&activations[l - 1].t());
                gb[l - 1] = delta.to_owned();
            }
            (gw, gb)
        }

    pub fn update_network(&mut self, iter: usize, gw: &Vec<Array2<f64>>, gb: &Vec<Array2<f64>>, n: usize, eta: f64, regularization: &Regularization) {
        let n_layers = self.sizes.len();
        for i in 0..n_layers - 1 {
            match self.optimizer {
                Optimizer::SGD => {
                    match regularization {
                        Regularization::None => {},
                        Regularization::L1(lambda) => {
                            self.weights[i].par_map_inplace(|x| {
                                let update = *x - (eta * lambda / n as f64) * x.signum();
                                *x = update;
                            })
                        },
                        Regularization::L2(lambda) => {
                            self.weights[i].par_map_inplace(|x| {
                                let update = 1.0 - eta * lambda / n as f64;
                                *x = *x * update;
                            })
                        },
                    }
                    self.weights[i] = &self.weights[i] - eta * &gw[i];
                    self.biases[i] = &self.biases[i] - eta * &gb[i];
                },
                Optimizer::Adam(AdamOptimizer {beta1, beta2, epsilon}) => {
                    self.w_epsilon1[i] = beta1 * &self.w_epsilon1[i] + (1.0 - beta1) * &gw[i];
                    self.b_epsilon1[i] = beta1 * &self.b_epsilon1[i] + (1.0 - beta1) * &gb[i];
                    self.w_epsilon2[i] = beta2 * &self.w_epsilon2[i] + (1.0 - beta2) * &gw[i].mapv(|x| x.powi(2));
                    self.b_epsilon2[i] = beta2 * &self.b_epsilon2[i] + (1.0 - beta2) * &gb[i].mapv(|x| x.powi(2));

                    let w_epsilon1_hat = &self.w_epsilon1[i] / (1.0 - beta1.powi(iter as i32));
                    let b_epsilon1_hat = &self.b_epsilon1[i] / (1.0 - beta1.powi(iter as i32));
                    let w_epsilon2_hat = &self.w_epsilon2[i] / (1.0 - beta2.powi(iter as i32));
                    let b_epsilon2_hat = &self.b_epsilon2[i] / (1.0 - beta2.powi(iter as i32));
                    
                    match regularization {
                        Regularization::None => {},
                        Regularization::L1(lambda) => {
                            self.weights[i].par_map_inplace(|x| {
                                let update = *x - (eta * lambda / n as f64) * x.signum();
                                *x = update;
                            })
                        },
                        Regularization::L2(lambda) => {
                            self.weights[i].par_map_inplace(|x| {
                                let update = 1.0 - eta * lambda / n as f64;
                                *x = *x * update;
                            })
                        },
                    }
                    self.weights[i] = &self.weights[i] - eta * &w_epsilon1_hat / (w_epsilon2_hat.mapv(f64::sqrt) + epsilon);
                    self.biases[i] = &self.biases[i] - eta * &b_epsilon1_hat / (b_epsilon2_hat.mapv(f64::sqrt) + epsilon);
                },
            }
        }
    }
}