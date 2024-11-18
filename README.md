A Julia implementation of a Convolutional Neural Network (CNN) for classifying CIFAR10 images using Flux.jl.
## Snippet Of Notebook (scroll for readme guide below)
![image](https://github.com/user-attachments/assets/d5cdbd93-b9b4-422a-a42e-9befdc7d9e97)

![image](https://github.com/user-attachments/assets/3365cb2f-4e4d-4ce0-ab72-ca8dc480fbed)

0.657
0.665
0.665
0.643
0.658
0.656
0.652
0.66
0.651
0.646
0.657
0.659
0.666
0.654
0.651
0.653
0.656
0.655
0.663
0.671
0.651
0.658
0.663
0.662
0.675
0.66
0.655
0.669
0.665
0.652
0.646
0.665
0.65
0 653

![image](https://github.com/user-attachments/assets/cec041a0-5070-41be-98c8-e0711796cbe4)

![image](https://github.com/user-attachments/assets/199d5d54-f21e-4f77-8259-19071203a1b4)

![image](https://github.com/user-attachments/assets/9f75a916-8f25-4c73-8fd4-8ca4589888c2)

![image](https://github.com/user-attachments/assets/37e01a66-ebea-4ec5-9541-295c65247fe5)

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

Pkg.activate(joinpath(@__DIR__, ".."))
OR 
Pkg.activate(".") //you will need to move the file

Pkg.instantiate()

The main notebook file can be found in notebooks/CIFAR10ImageClassification.jl
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
