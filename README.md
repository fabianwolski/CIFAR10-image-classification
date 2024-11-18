A Julia implementation of a Convolutional Neural Network (CNN) for classifying CIFAR10 images using Flux.jl.

## Features
- Implements a CNN architecture for image classification
- Supports GPU acceleration using CUDA.jl
- Achieves ~67% accuracy on the CIFAR10 test set
- Includes visualization tools for model performance analysis

## Requirements
- Julia 1.9+
- Flux.jl
- CUDA.jl
- MLDatasets
- Other dependencies listed in Project.toml

## Installation
1. Clone the repository:
```bash
git clone https://github.com/fabianwolski/CIFAR10-image-classification.git
cd CIFAR10-image-classification
```

2. Install dependencies:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Usage
Run the main training script:
```julia
include("src/CIFAR10ImageClassification.jl")
using .CIFAR10ImageClassification
train_model()
```

Or use the Pluto notebook:
```julia
using Pluto
Pluto.run(notebook="notebooks/cifar10_classification.jl")
```

## Model Architecture
The CNN architecture consists of:
- Multiple convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Dense layers for final classification
- Softmax output layer for 10-class classification

## Results
- Training accuracy: ~67%
- Test accuracy: ~67%
- Per-class performance varies (see detailed results in notebook)