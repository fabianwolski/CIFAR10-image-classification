export train_model, evaluate_model

"""
    train_model(model, train_data, val_data; epochs=100, learning_rate=0.01)

Train the model on CIFAR10 data.
"""
function train_model(model, train_data, val_data; epochs=100, learning_rate=0.01)
    loss(x, y) = Flux.crossentropy(model(x), y)
    optimizer = Momentum(learning_rate)
    
    train_history = Float32[]
    
    for epoch in 1:epochs
        # Training
        for d in train_data
            gs = gradient(Flux.params(model)) do
                l = loss(d...)
                return l
            end
            Flux.update!(optimizer, Flux.params(model), gs)
        end
        
        # Validation
        acc = evaluate_model(model, val_data)
        push!(train_history, acc)
        println("Epoch $epoch: Validation accuracy = $(round(acc, digits=3))")
    end
    
    return train_history
end

"""
    evaluate_model(model, data)

Evaluate model accuracy on given data.
"""
function evaluate_model(model, data)
    accuracy(x, y) = mean(onecold(model(x), 0:9) .== onecold(y, 0:9))
    
    total_acc = 0.0f0
    n_batches = length(data)
    
    for batch in data
        batch_acc = accuracy(batch...)
        total_acc += batch_acc
    end
    
    return total_acc / n_batches
end

"""
    evaluate_per_class(model, test_data)

Calculate per-class accuracy on test data.
"""
function evaluate_per_class(model, test_data)
    n_classes = 10
    class_correct = zeros(Int, n_classes)
    class_total = zeros(Int, n_classes)
    
    for (x_batch, y_batch) in test_data
        preds = cpu(model(x_batch))
        y_batch = cpu(y_batch)
        
        pred_classes = onecold(preds, 0:9)
        true_classes = onecold(y_batch, 0:9)
        
        for (pred, actual) in zip(pred_classes, true_classes)
            actual_index = actual + 1
            if pred == actual
                class_correct[actual_index] += 1
            end
            class_total[actual_index] += 1
        end
    end
    
    return class_correct ./ class_total
end