# Conditional Variational Autoencoder (CVAE) for MNIST

Implementation of CVAE for generating and analyzing MNIST digits with controllable digit labels.

## Features
- CVAE architecture with customizable latent dimensions
- Label conditioning via concatenation
- Multiple loss functions (BCE, MSE) comparison
- GPU support for M1/M2 MacBooks (MPS)
- Visualization tools

## Results
- BCE loss outperformed MSE for digit generation
- Latent space visualization shows digit clusters
- Successful conditional generation of specified digits

## Model Architecture
- Encoder: Input(784+10) → 512 → 256 → latent_dim
- Decoder: (latent_dim+10) → 256 → 512 → 784
- Label conditioning via one-hot encoding concatenation

## Training Details
- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 128
- Epochs: 10
- Loss: BCE + KL divergence

## Analysis Tools
- Loss curve plotting
- Latent space visualization
- Conditional sampling
- Multiple sample generation per digit

## Dependencies
- PyTorch
- torchvision
- matplotlib
- numpy

## Future Work
- Add quantitative metrics (FID score)
- Implement latent space interpolation
- Test different architectures
