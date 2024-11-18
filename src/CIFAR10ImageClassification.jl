module CIFAR10ImageClassification

using Flux
using CUDA
using MLDatasets
using Statistics
using Images.ImageCore
using Flux: onehotbatch, onecold
using Base.Iterators: partition

export train_model, evaluate_model, visualize_results

include("data.jl")
include("model.jl")
include("train.jl")
include("utils.jl")

end # module