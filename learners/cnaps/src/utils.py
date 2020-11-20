import torch
import torch.nn.functional as F
import os
import math
from enum import Enum
import sys


class ValidationAccuracies:
    """
    Determines if an evaluation on the validation set is better than the best so far.
    In particular, this handles the case for meta-dataset where we validate on multiple datasets and we deem
    the evaluation to be better if more than half of the validation accuracies on the individual validation datsets
    are better than the previous best.
    """

    def __init__(self, validation_datasets):
        self.datasets = validation_datasets
        self.dataset_count = len(self.datasets)
        self.current_best_accuracy_dict = {}
        for dataset in self.datasets:
            self.current_best_accuracy_dict[dataset] = {"accuracy": 0.0, "confidence": 0.0}

    def is_better(self, accuracies_dict):
        is_better = False
        is_better_count = 0
        for i, dataset in enumerate(self.datasets):
            if accuracies_dict[dataset]["accuracy"] > self.current_best_accuracy_dict[dataset]["accuracy"]:
                is_better_count += 1

        if is_better_count >= int(math.ceil(self.dataset_count / 2.0)):
            is_better = True

        return is_better

    def replace(self, accuracies_dict):
        self.current_best_accuracy_dict = accuracies_dict

    def print(self, logfile, accuracy_dict):
        print_and_log(logfile, "")  # add a blank line
        print_and_log(logfile, "Validation Accuracies:")
        for dataset in self.datasets:
            print_and_log(logfile, "{0:}: {1:.1f}+/-{2:.1f}".format(dataset, accuracy_dict[dataset]["accuracy"],
                                                                    accuracy_dict[dataset]["confidence"]))
        print_and_log(logfile, "")  # add a blank line

    def get_current_best_accuracy_dict(self):
        return self.current_best_accuracy_dict


def verify_checkpoint_dir(checkpoint_dir, resume, test_mode):
    if resume:  # verify that the checkpoint directory and file exists
        if not os.path.exists(checkpoint_dir):
            print("Can't resume for checkpoint. Checkpoint directory ({}) does not exist.".format(checkpoint_dir), flush=True)
            sys.exit()

        checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pt')
        if not os.path.isfile(checkpoint_file):
            print("Can't resume for checkpoint. Checkpoint file ({}) does not exist.".format(checkpoint_file), flush=True)
            sys.exit()
    elif test_mode:
        if not os.path.exists(checkpoint_dir):
            print("Can't test. Checkpoint directory ({}) does not exist.".format(checkpoint_dir), flush=True)
            sys.exit()
    else:
        if os.path.exists(checkpoint_dir):
            print("Checkpoint directory ({}) already exits.".format(checkpoint_dir), flush=True)
            print("If starting a new training run, specify a directory that does not already exist.", flush=True)
            print("If you want to resume a training run, specify the -r option on the command line.", flush=True)
            sys.exit()

def write_to_log(log_file, message):
    log_file.write(message + "\n")

def print_and_log(log_file, message):
    """
    Helper function to print to the screen and the cnaps_layer_log.txt file.
    """
    print(message, flush=True)
    log_file.write(message + '\n')


def get_log_files(checkpoint_dir, resume, test_mode):
    """
    Function that takes a path to a checkpoint directory and returns a reference to a logfile and paths to the
    fully trained model and the model with the best validation score.
    """
    verify_checkpoint_dir(checkpoint_dir, resume, test_mode)
    if not test_mode and not resume:
        os.makedirs(checkpoint_dir)
    checkpoint_path_validation = os.path.join(checkpoint_dir, 'best_validation.pt')
    checkpoint_path_final = os.path.join(checkpoint_dir, 'fully_trained.pt')
    logfile_path = os.path.join(checkpoint_dir, 'log.txt')
    if os.path.isfile(logfile_path):
        logfile = open(logfile_path, "a", buffering=1)
    else:
        logfile = open(logfile_path, "w", buffering=1)
    debugfile_path = os.path.join(checkpoint_dir, 'dump.txt')
    if os.path.isfile(debugfile_path):
        debugfile = open(debugfile_path, "a", buffering=1)
    else:
        debugfile = open(debugfile_path, "w", buffering=1)

    return checkpoint_dir, logfile, checkpoint_path_validation, checkpoint_path_final, debugfile


def stack_first_dim(x):
    """
    Method to combine the first two dimension of an array
    """
    x_shape = x.size()
    new_shape = [x_shape[0] * x_shape[1]]
    if len(x_shape) > 2:
        new_shape += x_shape[2:]
    return x.view(new_shape)


def split_first_dim_linear(x, first_two_dims):
    """
    Undo the stacking operation
    """
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += [x_shape[-1]]
    return x.view(new_shape)


def sample_normal(mean, var, num_samples):
    """
    Generate samples from a reparameterized normal distribution
    :param mean: tensor - mean parameter of the distribution
    :param var: tensor - variance of the distribution
    :param num_samples: np scalar - number of samples to generate
    :return: tensor - samples from distribution of size numSamples x dim(mean)
    """
    sample_shape = [num_samples] + len(mean.size())*[1]
    normal_distribution = torch.distributions.Normal(mean.repeat(sample_shape), var.repeat(sample_shape))
    return normal_distribution.rsample()


def loss(test_logits_sample, test_labels, device):
    """
    Compute the classification loss.
    """
    size = test_logits_sample.size()
    sample_count = size[0]  # scalar for the loop counter
    num_samples = torch.tensor([sample_count], dtype=torch.float, device=device, requires_grad=False)

    log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=device)
    for sample in range(sample_count):
        log_py[sample] = -F.cross_entropy(test_logits_sample[sample], test_labels, reduction='none')
    score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)
    return -torch.sum(score, dim=0)


def aggregate_accuracy(test_logits_sample, test_labels):
    """
    Compute classification accuracy.
    """
    averaged_predictions = torch.logsumexp(test_logits_sample, dim=0)
    return torch.mean(torch.eq(test_labels, torch.argmax(averaged_predictions, dim=-1)).float())


def linear_classifier(x, param_dict):
    """
    Classifier.
    """
    return F.linear(x, param_dict['weight_mean'], param_dict['bias_mean'])


def mlpip_loss(test_logits_samples, test_labels, device):
    """
    Compute the ML-PIP loss given sample logits, and test labels. Computes as
        log (1 / L) ∑ exp ( log p(D_t | z_l) )
            = log ∑ exp ( log p(D_t | z_l) ) - log L
            = LSE_l (log p(D_t | z_l)) - log L

    Args:
        test_logits_samples (torch.tensor): samples of logits for forward pass
        (num_samples x num_target x num_classes)
        test_labels (torch.tensor): ground truth labels
        (num_target x num_classes)
        device (torch.device): device we're running on
    Returns:
        (torch.scalar): computation of ML-PIP loss function
    """
    # Extract number of samples
    num_samples = test_logits_samples.shape[0]

    # log p(D_t | z_l) = ∑ log p(y | x, z_l)
    pyx = torch.distributions.categorical.Categorical(logits=test_logits_samples)
    log_py = pyx.log_prob(test_labels).sum(dim=1)
    # tensorize L
    l = torch.tensor([num_samples], dtype=torch.float, device=device)
    mlpips = torch.logsumexp(log_py, dim=0) - torch.log(l)
    return -mlpips.mean()


def sample_normal(mean, var, num_samples):
    """
    Generate samples from a reparameterized normal distribution
    :param mean: tf tensor - mean parameter of the distribution
    :param var: tf tensor - log variance of the distribution
    :param num_samples: np scalar - number of samples to generate
    :return: tf tensor - samples from distribution of size numSamples x dim(mean)
    """
    # example: sample_shape = [L, 1, 1, 1, 1]
    sample_shape = [num_samples] + len(mean.size())*[1]
    normal_distribution = torch.distributions.Normal(mean.repeat(sample_shape), var.repeat(sample_shape))
    return normal_distribution.rsample()


def mlpip_classifier(x, param_dict, num_samples):
    mean = F.linear(x, param_dict['weight_mean'], param_dict['bias_mean'])
    var = F.linear(x**2, param_dict['weight_variance'], param_dict['bias_variance'])
    sample = sample_normal(mean, var, num_samples)
    return sample


def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector



