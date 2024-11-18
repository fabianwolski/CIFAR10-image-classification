export visualize_sample, plot_training_history, display_class_accuracies

using Plots
using Images.ImageCore

"""
    visualize_sample(x, pred_label, true_label)

Display a sample image with its predicted and true labels.
"""
function visualize_sample(x, pred_label, true_label)
    # Convert array to image
    img = colorview(RGB, permutedims(x, (3, 2, 1)))
    
    # Create title based on prediction correctness
    title = pred_label == true_label ? "$pred_label ✓" : "$pred_label (true: $true_label)"
    
    plot(img, title=title, axis=nothing)
end

"""
    plot_training_history(history)

Plot training history showing accuracy over epochs.
"""
function plot_training_history(history)
    plot(history,
         ylim=(0.0, 1.0),
         legend=false,
         title="Training Accuracy",
         xlabel="Epoch",
         ylabel="Accuracy",
         linewidth=2)
end

"""
    display_class_accuracies(accuracies)

Display per-class accuracies in a formatted way.
"""
function display_class_accuracies(accuracies)
    for (acc, class) in zip(accuracies, CLASSES)
        println("$class: $(round(acc * 100, digits=1))%")
    end
end