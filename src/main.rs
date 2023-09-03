use candle_core::{DType, Device, Result, Tensor};

struct Model {
    first: Tensor,
    second: Tensor,
}


impl Model {
    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = image.matmul(&self.first)?;
        let x = x.relu()?;
        x.matmul(&self.second)
    }
}

fn main() -> Result<()> {
   
// use Device::new_cuda(0)?; to use the GPU.
let device = match Device::new_cuda(0) {
    Ok(device) => device,
    Err(_) => Device::Cpu,
};   




let first = Tensor::zeros((784, 100).contiguous()?, DType::F32, &device)?;
let second = Tensor::zeros((100, 10).contiguous()?, DType::F32, &device)?;
let model = Model {first, second};

let dummy_image = Tensor::zeros((1, 784), DType::F32, &matched_device)?;

let digit = model.forward(&dummy_image)?;

println!("Digit {digit:?} digit");

Ok(())
}

