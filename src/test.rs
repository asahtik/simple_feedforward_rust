use ndarray::prelude::*;

mod network;
use network::*;

fn bench1() {
    let mut net = Network::new(vec![3,2,2], Optimizer::SGD);
    net.weights = vec![
        array![[0.1, 0.2, 0.5], [0.4, 0.1, 0.5]],
        array![[0.6, 0.1], [0.5, 0.2]]
    ];
    net.biases = vec![
        array![[0.2], [0.2]],
        array![[0.2], [0.1]]
    ];
    let inputs = array![0.5, 0.5, 0.1];
    let targets = array![0.9, 0.1];

    let (outputs, z_s, a_s) = 
        net.forward_pass(&inputs.view());

    let (gw, gb) = 
        net.backward_pass(&outputs.view(), &targets.view(), &z_s, &a_s);
    
    net.update_network(1, &gw, &gb, 1, 0.01, &Regularization::None);

    println!("Cost: {}", cost(&outputs.view(), &targets.view()));
}