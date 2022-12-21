import os
import torch
import torch.nn.functional as F


class Logger():
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


def accuracy(logits, test_labels):
    """
    Compute classification accuracy.
    """
    return torch.mean(torch.eq(test_labels, torch.argmax(logits, dim=-1)).float())


def loss_fn(logits, labels):
    return F.cross_entropy(logits, labels)


def set_pytorch_seeds():
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
