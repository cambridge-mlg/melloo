import os
import sys
import torch


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

    return checkpoint_dir, logfile, checkpoint_path_validation, checkpoint_path_final


def categorical_accuracy(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def euclidean_metric(target_features, class_prototypes):
    num_target_features = target_features.shape[0]
    num_prototypes = class_prototypes.shape[0]

    distances = (target_features.unsqueeze(1).expand(num_target_features, num_prototypes, -1) -
                 class_prototypes.unsqueeze(0).expand(num_target_features, num_prototypes, -1)).pow(2).sum(dim=2)
    return -distances


def loss(logits, labels):
    return torch.nn.functional.cross_entropy(logits, labels)


