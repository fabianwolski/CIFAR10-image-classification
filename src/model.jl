export create_model, create_model_with_extra_conv, create_model_with_extra_dense

"""
    create_model()

Create base CNN model for CIFAR10 classification.
"""
function create_model()
    return Chain(
        Conv((5,5), 3=>16, pad=SamePad(), relu),
        MaxPool((2,2)),
        Conv((5,5), 16=>8, pad=SamePad(), relu),
        MaxPool((2,2)),
        Flux.flatten,
        Dense(512, 256),
        Dense(256, 10),
        softmax) |> gpu
end

"""
    create_model_with_extra_conv()

Create CNN model with additional convolutional layer.
"""
function create_model_with_extra_conv()
    return Chain(
        Conv((5,5), 3=>16, pad=SamePad(), relu),
        MaxPool((2,2)),
        Conv((5,5), 16=>16, pad=SamePad(), relu),
        MaxPool((2,2)),
        Conv((5,5), 16=>8, pad=SamePad(), relu),
        MaxPool((2,2)),
        Flux.flatten,
        Dense(512, 256),
        Dense(256, 10),
        softmax) |> gpu
end

"""
    create_model_with_extra_dense()

Create CNN model with additional dense layer.
"""
function create_model_with_extra_dense()
    return Chain(
        Conv((5,5), 3=>16, pad=SamePad(), relu),
        MaxPool((2,2)),
        Conv((5,5), 16=>8, pad=SamePad(), relu),
        MaxPool((2,2)),
        Flux.flatten,
        Dense(512, 256),
        Dense(256, 128),
        Dense(128, 10),
        softmax) |> gpu
end