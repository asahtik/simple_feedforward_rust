extern crate blas_src;

use ndarray::Axis;
use network::{AdamOptimizer};
use pyo3::{prelude::*};
use numpy::{PyReadonlyArray2, PyReadonlyArray1, PyArray1, IntoPyArray};

mod network;

#[pymodule]
fn ff_network(_: Python, m: &PyModule) -> PyResult<()> {
    #[pyclass]
    struct Network {
        nn: Box<network::Network>,
        regularization: Box<network::Regularization>
    }

    #[pymethods]
    impl Network {
        #[new]
        fn py_new(sizes: Vec<usize>) -> Self {
            Self { 
                nn: Box::new(network::Network::new(sizes, 
                network::Optimizer::Adam(AdamOptimizer {
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8
                }))),
                regularization: Box::new(network::Regularization::None)
            }
        }

        fn set_optimizer_sgd(&mut self) {
            self.nn.optimizer = network::Optimizer::SGD;
        }

        fn set_optimizer_adam(&mut self, beta1: f64, beta2: f64, epsilon: f64) {
            self.nn.optimizer = network::Optimizer::Adam(network::AdamOptimizer {
                beta1,
                beta2,
                epsilon
            })
        }

        fn set_regularization_none(&mut self) {
            self.regularization = 
                Box::new(network::Regularization::None);
        }

        fn set_regularization_l1(&mut self, lambda: f64) {
            self.regularization = 
                Box::new(network::Regularization::L1(lambda));
        }

        fn set_regularization_l2(&mut self, lambda: f64) {
            self.regularization = 
                Box::new(network::Regularization::L2(lambda));
        }

        fn set_weights(&mut self, ws: Vec<PyReadonlyArray2<f64>>) {
            let n = self.nn.weights.len();
            if n == ws.len() {
                for i in 0..n {
                    let w = ws[i].as_array().to_owned();
                    if self.nn.weights[i].shape() == w.shape() {
                        self.nn.weights[i].clone_from(&w);
                    } else {
                        panic!("Incorrect shape");
                    }
                }
            } else {
                panic!("Incorrect length");
            }
        }

        fn set_biases(&mut self, bs: Vec<PyReadonlyArray1<f64>>) {
            let n = self.nn.biases.len();
            if n == bs.len() {
                for i in 0..n {
                    let b = bs[i].as_array().to_owned();
                    let b = b.insert_axis(Axis(1));
                    if self.nn.biases[i].shape() == b.shape() {
                        self.nn.biases[i].clone_from(&b);
                    } else {
                        panic!("Incorrect shape");
                    }
                }
            } else {
                panic!("Incorrect length");
            }
        }

        fn train(&mut self, training_data: PyReadonlyArray2<f64>, 
            training_class: PyReadonlyArray2<f64>, val_data: PyReadonlyArray2<f64>, 
            val_class: PyReadonlyArray2<f64>, epochs: usize, 
            mini_batch_size: usize, eta: f64, decay_rate: f64) {
                self.nn.as_mut().train(&training_data.as_array(), &training_class.as_array(), &val_data.as_array(), 
                    &val_class.as_array(), epochs, mini_batch_size, eta, decay_rate, &self.regularization);
            }

        fn evaluate(&self, val_data: PyReadonlyArray2<f64>, val_class: PyReadonlyArray2<f64>) {
            self.nn.eval_network(&val_data.as_array(), &val_class.as_array());
        }

        fn predict<'py>(&self, data: PyReadonlyArray1<f64>, py: Python<'py>) -> &'py PyArray1<f64> {
            let output = self.nn.predict(&data.as_array());
            output.into_pyarray(py)
        }
    }
    m.add_class::<Network>()?;
    Ok(())
}