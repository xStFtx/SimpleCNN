use tch::{nn, nn::ModuleT, Device, Tensor, nn::OptimizerConfig, nn::functional as F};
use tch::nn::{Conv2D, Linear};
use tch::vision::datasets::{mnist, MNIST};
use tch::nn::optim::Adam;
use tch::data::transforms::{Resize, Normalize, Grayscale};
use tch::autograd::Variable;

struct SimpleCNN {
    conv1: Conv2D,
    conv2: Conv2D,
    fc1: Linear,
    fc2: Linear,
}

impl SimpleCNN {
    fn new(vs: nn::Path) -> SimpleCNN {
        let conv1 = nn::conv2d(vs / "conv1", 1, 32, 5, Default::default());
        let conv2 = nn::conv2d(vs / "conv2", 32, 64, 5, Default::default());
        let fc1 = nn::linear(vs / "fc1", 64 * 4 * 4, 128, Default::default());
        let fc2 = nn::linear(vs / "fc2", 128, 10, Default::default());

        SimpleCNN {
            conv1,
            conv2,
            fc1,
            fc2,
        }
    }
}

impl nn::ModuleT for SimpleCNN {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let x = xs.view([-1, 1, 28, 28]);

        let x = F::relu(&self.conv1.forward(&x));
        let x = F::max_pool2d(&x, &[2, 2], &[2, 2], &[0, 0], &[1, 1], false);

        let x = F::relu(&self.conv2.forward(&x));
        let x = F::max_pool2d(&x, &[2, 2], &[2, 2], &[0, 0], &[1, 1], false);

        let x = x.view([-1, 64 * 4 * 4]);
        let x = F::relu(&self.fc1.forward(&x));
        let x = F::dropout(&x, 0.5, train);
        self.fc2.forward(&x)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let net = SimpleCNN::new(vs.root());

    let train = mnist("../data")
        .train()?
        .shuffle(64)
        .transform(trn());

    let mut opt = Adam::default().build(&net, 1e-3)?;

    for epoch in 1..100 {
        let loss = net.batch_accuracy(&train)?;
        opt.backward_step(&loss);
        println!("epoch: {:4} loss: {:8.5}", epoch, f64::from(&loss));
    }

    Ok(())
}

fn trn() -> impl Fn(&Tensor) -> Tensor {
    transforms::compose(vec![
        Resize([28, 28]),
        Normalize::default(),
        Grayscale::default(),
    ])
}
