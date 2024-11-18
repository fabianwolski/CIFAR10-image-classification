#data loading and preprocessing functions
export load_cifar10, prepare_data

const CLASSES = ["airplane", "automobile", "bird", "cat",
                "deer", "dog", "frog", "horse", "ship", "truck"]

"""
    load_cifar10()

Load and preprocess CIFAR10 dataset.
"""
function load_cifar10()
    train_x, train_y = CIFAR10(split=:train)[:]
    test_x, test_y = CIFAR10(split=:test)[:]
    
    # Convert labels to one-hot encoding
    train_labels = onehotbatch(train_y, 0:9)
    test_labels = onehotbatch(test_y, 0:9)
    
    return (train_x, train_labels), (test_x, test_labels)
end

"""
    prepare_data(x_data, labels; batch_size=1000)

Prepare data for training by reshaping and batching.
"""
function prepare_data(x_data, labels; batch_size=1000)
    # Reshape data for convolution
    x_reshaped = reshape(x_data, 32, 32, 3, :)
    
    # Create batches
    batches = [(x_reshaped[:,:,:,i:i+batch_size-1], 
                labels[:,i:i+batch_size-1])
              for i in 1:batch_size:size(x_reshaped, 4)-batch_size+1]
    
    return batches |> gpu
end

"""
    split_validation(train_data, train_labels; val_size=1000)

Split training data into training and validation sets.
"""
function split_validation(train_data, train_labels; val_size=1000)
    n_samples = size(train_data, 4)
    val_indices = n_samples-val_size+1:n_samples
    train_indices = 1:n_samples-val_size
    
    train_set = prepare_data(train_data[:,:,:,train_indices], 
                            train_labels[:,train_indices])
    val_set = prepare_data(train_data[:,:,:,val_indices], 
                          train_labels[:,val_indices])
    
    return train_set, val_set
end