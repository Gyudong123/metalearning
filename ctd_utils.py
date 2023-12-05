import numpy as np
import torch

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def sum_params_across_models(params_lists):
    """
    Given a list of model parameters lists (e.g., [W1, W2, ...]),
    compute the sum of corresponding parameters.
    """
    # Use zip to group corresponding parameters across all models
    summed_params = [torch.sum(torch.stack(ps), dim=0) for ps in zip(*params_lists)]
    return summed_params

def average_params_across_models(params_lists):
    """
    Given a list of model parameters lists (e.g., [W_before1, W_before2, ...]),
    compute the average of corresponding parameters.
    """
    # Use zip to group corresponding parameters across all models
    avg_params = [torch.mean(torch.stack(ps), dim=0) for ps in zip(*params_lists)]
    return avg_params

def fast_adapt_train(meta_c, task_c, batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error, meta_c, task_c)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy

