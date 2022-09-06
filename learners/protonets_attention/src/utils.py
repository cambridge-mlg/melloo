import torch
import torch.nn.functional as F
import os
import math
import sys
import numpy as np
from enum import Enum
from PIL import Image


class MetaLearningState(Enum):
    META_TRAIN = 0
    META_TEST = 1


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

    def print(self, logger, accuracy_dict):
        logger.print_and_log("")  # add a blank line
        logger.print_and_log("Validation Accuracies:")
        for dataset in self.datasets:
            logger.print_and_log("{0:}: {1:.1f}+/-{2:.1f}".format(dataset, accuracy_dict[dataset]["accuracy"],
                                                                    accuracy_dict[dataset]["confidence"]))
        logger.print_and_log("")  # add a blank line

    def get_current_best_accuracy_dict(self):
        return self.current_best_accuracy_dict


class LogFiles:
    def __init__(self, checkpoint_dir, resume, test_mode):
        self._checkpoint_dir = checkpoint_dir
        if not self._verify_checkpoint_dir(resume, test_mode):
            sys.exit()
        if not test_mode and not resume:
            os.makedirs(self.checkpoint_dir)
        self._best_validation_model_path = os.path.join(checkpoint_dir, 'best_validation.pt')
        self._fully_trained_model_path = os.path.join(checkpoint_dir, 'fully_trained.pt')

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @property
    def best_validation_model_path(self):
        return self._best_validation_model_path

    @property
    def fully_trained_model_path(self):
        return self._fully_trained_model_path

    def _verify_checkpoint_dir(self, resume, test_mode):
        checkpoint_dir_is_ok = True
        if resume:  # verify that the checkpoint directory and file exists
            if not os.path.exists(self.checkpoint_dir):
                print("Can't resume from checkpoint. Checkpoint directory ({}) does not exist.".format(self.checkpoint_dir), flush=True)
                checkpoint_dir_is_ok = False

            checkpoint_file = os.path.join(self.checkpoint_dir, 'checkpoint.pt')
            if not os.path.isfile(checkpoint_file):
                print("Can't resume for checkpoint. Checkpoint file ({}) does not exist.".format(checkpoint_file), flush=True)
                checkpoint_dir_is_ok = False

        elif test_mode:
            if not os.path.exists(self.checkpoint_dir):
                print("Can't test. Checkpoint directory ({}) does not exist.".format(self.checkpoint_dir), flush=True)
                checkpoint_dir_is_ok = False

        else:
            if os.path.exists(self.checkpoint_dir):
                print("Checkpoint directory ({}) already exits.".format(self.checkpoint_dir), flush=True)
                print("If starting a new training run, specify a directory that does not already exist.", flush=True)
                print("If you want to resume a training run, specify the -r option on the command line.", flush=True)
                checkpoint_dir_is_ok = False

        return checkpoint_dir_is_ok


class Logger:
    def __init__(self, checkpoint_dir, log_file_name):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        log_file_path = os.path.join(checkpoint_dir, log_file_name)
        self.file = None
        if os.path.isfile(log_file_path):
            self.file = open(log_file_path, "a", buffering=1)
        else:
            self.file = open(log_file_path, "w", buffering=1)

    def __del__(self):
        self.file.close()

    def log(self, message):
        self.file.write(message + '\n')

    def print_and_log(self, message):
        print(message, flush=True)
        self.log(message)


def linear_classifier(x, param_dict):
    """
    Classifier.
    """
    return F.linear(x, param_dict['weight_mean'], param_dict['bias_mean'])


def cross_transformer_linear_classifier(x, param_dict):
    """
    Classifier.
    """
    bias = param_dict['bias_mean'].permute(0,2,1)
    weights = param_dict['weight_mean'].permute(0,2,1)
    return (torch.baddbmm(beta=1.0, alpha=1.0, input=bias, batch1=x, batch2=weights)).squeeze(dim=1)


def attentive_linear_classifier(x, param_dict):
    """
    Classifier.
    """
    bias = param_dict['bias_mean'].permute(0,2,1)
    features = torch.unsqueeze(x, dim=1)
    weights = param_dict['weight_mean'].permute(0,2,1)
    return torch.baddbmm(beta=1.0, alpha=1.0, input=bias, batch1=features, batch2=weights)


def cross_entropy_loss(logits, labels, reduce=True):
    return F.cross_entropy(logits, labels, reduce=reduce)


# Each centroid/class cluster has a mu and sigma.
# The size of the final embedding layer determines the dimension of mu
def kl_divergence(mu1, prec1, mu2, prec2):
    sigma1 = torch.inverse(prec1)
    return 0.5 * (
        (  (mu2 - mu1).T @ prec2 @ (mu2 - mu1)
        + (prec2 @ sigma1).trace()
        - torch.log(torch.det(prec2)/torch.det(prec1))  ) # det(A) = 1/deta(A^-1)
        .add(mu1.shape[0], -1)
    )

def coalesce_labels(labels, mode="by_example"):
    if mode == "by_class":
        labels = torch.unique(labels)

    return labels


def categorical_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=1)
    return torch.mean(torch.eq(predictions, labels).float())


def extract_class_indices(labels, which_class):
    if isinstance(labels, torch.Tensor):
       class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
       class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
       return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector
    else:
        if isinstance(which_class, torch.Tensor):
            which_class = which_class.item()
        return np.argwhere(labels == which_class).reshape(-1)


def merge_logits(logits, labels, method):
    if method == "sum_by_class":
        class_probabilities_sum = []
        for c in torch.unique(labels):
            class_logits = torch.index_select(logits, 0, extract_class_indices(labels, c))
            class_probabilities = F.softmax(class_logits, dim=-1)
            class_probabilities_sum.append(torch.sum(class_probabilities, dim=0))
        return torch.stack(class_probabilities_sum)
    elif method == "product_by_class":
        class_probabilities_product = []
        for c in torch.unique(labels):
            class_logits = torch.index_select(logits, 0, extract_class_indices(labels, c))
            log_class_probabilities = F.log_softmax(class_logits, dim=-1)
            class_probabilities_product.append(torch.sum(log_class_probabilities, dim=0))
        return torch.stack(class_probabilities_product)
    else:
        return logits  # do nothing


def mode_accuracy(logits, labels):
    accuracies = []
    predictions = torch.argmax(logits, dim=1)
    for c in torch.unique(labels):
        class_predictions = torch.index_select(predictions, 0, extract_class_indices(labels, c))
        values, _ = torch.mode(class_predictions)
        accuracies.append(torch.eq(values, c).float())
    merged_accuracies = torch.stack(accuracies)

    return torch.mean(merged_accuracies)


def save_image(image_array, save_path):
    image_array = image_array.squeeze()
    image_array = image_array.transpose([1, 2, 0])
    im = Image.fromarray(np.clip((image_array + 1.0) * 127.5 + 0.5, 0, 255).astype(np.uint8), mode='RGB')
    im.save(save_path)

# Have updated with version in split_wrapper
def convert_to_array(my_list):
    if type(my_list) == np.int32 or type(my_list) == int:
        return np.array([my_list])
    return my_list

def convert_to_numpy(x):
    if torch.is_tensor(x):
        if x.is_cuda:
            return x.cpu().numpy()
        else:
            return x.numpy()
    else:
        return x


def move_set_to_cuda(images, labels, device):
        if not images.is_cuda:
            images = images.to(device)
        if not labels.is_cuda:
            labels = labels.type(torch.LongTensor).to(device)
        return images, labels
        
def np_set_to_torch(images_np, labels_np, shuffle=False):
    if len(images_np.shape) == 3:
        images_np = images_np.unsqueeze(0)
    images_np = images_np.transpose([0, 3, 1, 2])
    # Don't shuffle so that we are assured consistency in the noisy shot detection case
    if shuffle:
        images_np, labels_np = shuffle_set(images_np, labels_np)
    images = torch.from_numpy(images_np)
    labels = torch.from_numpy(labels_np)
    return images, labels
    

def prepare_task(task_dict, device, shuffle=False):
    # If context_images are already a tensor, just assign
    context_images, context_labels = task_dict['context_images'], task_dict['context_labels']
    target_images, target_labels = task_dict['target_images'], task_dict['target_labels']
    # Else, assume numpy format and do conversion
    if not torch.is_tensor(context_images):
        context_images, context_labels = np_set_to_torch(context_images, context_labels, shuffle)
        target_images, target_labels = np_set_to_torch(target_images, target_labels, shuffle)

    context_images, context_labels = move_set_to_cuda(context_images, context_labels, device)
    target_images, target_labels = move_set_to_cuda(target_images, target_labels, device)
    return context_images, target_images, context_labels, target_labels

    
def shuffle_set(images, labels):
    """
    Return shuffled data.
    """
    permutation = np.random.permutation(images.shape[0])
    return images[permutation], labels[permutation]
