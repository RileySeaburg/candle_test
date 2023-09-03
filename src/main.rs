use candle_core::{DType, Device, Result, Tensor};


pub struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = input.matmul(&self.weight)?;
        x.broadcast_add(&self.bias)
    }
}

pub struct Model {
    first: Linear,
    second: Linear,
}


impl Model {
    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = self.first.forward(image)?;
        let x = x.relu()?;
        self.second.forward(&x)
    }
}

fn main() -> Result<()> {
    // use Device::new_cuda(0)?; to use the GPU.
    let device = match Device::new_cuda(0) {
        Ok(device) => device,
        Err(_) => Device::Cpu,
    };



    // setting first layer weights
    let weight = Tensor::zeros((784, 100), DType::F32, &device)
        .unwrap()
        .contiguous()?;

    // setting first layer bias
    let bias = Tensor::zeros((100,), DType::F32, &device)
        .unwrap()
        .contiguous()?;

    let first = Linear { weight, bias };
    
    // setting second layer weights
    let weight = Tensor::zeros((100, 10), DType::F32, &device)
        .unwrap()
        .contiguous()?;

    // setting second layer bias
    let bias = Tensor::zeros((10,), DType::F32, &device)
        .unwrap()
        .contiguous()?;

    let second = Linear { weight, bias };

    let model = Model { first, second };

    let dummy_image = Tensor::zeros((1, 784), DType::F32, &device)
        .unwrap()
        .contiguous()?;

    // Inference on the model
    let digit = model.forward(&dummy_image)?;

    println!("Digit {digit:?} digit");

    Ok(())
}
