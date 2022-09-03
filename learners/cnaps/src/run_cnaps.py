"""
Script to reproduce the few-shot classification results on Meta-Dataset in:
"Fast and Flexible Multi-Task Classification Using Conditional Neural Adaptive Processes"
https://arxiv.org/pdf/1906.07697.pdf

The following command lines should reproduce the published results within error-bars:

Note before running any of the commands, you need to run the following two commands:

ulimit -n 50000

export META_DATASET_ROOT=<root directory of the cloned or downloaded Meta-Dataset repository>

CNAPs using auto-regressive FiLM adaptation, meta-training on all datasets
--------------------------------------------------------------------------
python run_cnaps.py --data_path <path to directory containing Meta-Dataset records>

CNAPs using FiLM adaptation only, meta-training on all datasets
---------------------------------------------------------------
python run_cnaps.py --feature_adaptation film --data_path <path to directory containing Meta-Dataset records>

CNAPs using no feature adaptation, meta-training on all datasets
----------------------------------------------------------------
python run_cnaps.py --feature_adaptation no_adaptation --data_path <path to directory containing Meta-Dataset records>

CNAPs using FiLM adaptation and TaskNorm, meta-training on all datasets
-----------------------------------------------------------------------
python run_cnaps.py --feature_adaptation film -i 60000 -lr 0.001 --batch_normalization task_norm-i
                    --data_path <path to directory containing Meta-Dataset records>

- Note that when using Meta-Dataset and auto-regressive FiLM adaptation or FiLM adaptation with TaskNorm
batch normalization, 2 GPUs with at least 16GB of memory are required.
- The other modes require only a single GPU with at least 16 GB of memory.
- If you want to run any of the modes on a single GPU, you can train on a single dataset with fixed shot and way, an
example command line is (though this will not reproduce the meta-dataset results):
python run_cnaps.py --feature_adaptation film -i 20000 -lr 0.001 --batch_normalization task_norm-i
                    -- dataset omniglot --way 5 --shot 5 --data_path <path to directory containing Meta-Dataset records>

"""
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from normalization_layers import TaskNormI
from utils import print_and_log, write_to_log, get_log_files, ValidationAccuracies, loss, aggregate_accuracy, verify_checkpoint_dir, SavedDataset
from model import Cnaps
from meta_dataset_reader import MetaDatasetReader, SingleDatasetReader
from cifar_dataset import CIFAR, dataset_from_metdataset_task_dict

from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Quiet TensorFlow warnings

from PIL import Image
import sys

NUM_VALIDATION_TASKS = 200
NUM_TEST_TASKS = 600
PRINT_FREQUENCY = 1000
NUM_INDEP_EVAL_TASKS = 50



def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        if not os.path.exists(self.args.checkpoint_dir):
            os.makedirs(self.args.checkpoint_dir)

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final, self.debugfile \
            = get_log_files(self.args.checkpoint_dir, self.args.resume_from_checkpoint, self.args.mode == "test")

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        gpu_device = 'cuda:0'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.train_set, self.validation_set, self.test_set = self.init_data()

        if self.args.dataset == "meta-dataset":
            if self.args.query_test * self.args.target_set_size_multiplier > 50:
                print_and_log(self.logfile, "WARNING: Very high number of query points requested. Query points = query_test * target_set_size_multiplier = {} * {} = {}".format(self.args.query_test, self.args.target_set_size_multiplier, self.args.query_test * self.args.target_set_size_multiplier))

            self.dataset = MetaDatasetReader(self.args.data_path, self.args.mode, self.train_set, self.validation_set,
                                             self.test_set, self.args.max_way_train, self.args.max_way_test,
                                             self.args.max_support_train, self.args.max_support_test, self.args.query_test)
        elif self.args.dataset != "from_file":
            if self.args.construct_coreset:
                self.dataset = CIFAR(self.args.way, self.args.shot, self.args.query_test)
            else:
                self.dataset = SingleDatasetReader(self.args.data_path, self.args.mode, self.args.dataset, self.args.way,
                                                   self.args.shot, self.args.query_train, self.args.query_test)
        else:
            self.dataset = SavedDataset(self.args.data_path)

        self.loss = loss
        self.accuracy_fn = aggregate_accuracy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.validation_accuracies = ValidationAccuracies(self.validation_set)
        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()

    def init_model(self):
        use_two_gpus = self.use_two_gpus()
        model = Cnaps(device=self.device, use_two_gpus=use_two_gpus, args=self.args).to(self.device)
        self.register_extra_parameters(model)

        # set encoder is always in train mode (it only sees context data).
        model.train()
        # Feature extractor is in eval mode by default, but gets switched in model depending on args.batch_normalization
        model.feature_extractor.eval()
        if use_two_gpus:
            model.distribute_model()
        return model

    def init_data(self):
        if self.args.dataset == "meta-dataset":
            train_set = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower']
            validation_set = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi',
                              'vgg_flower',
                              'mscoco']
            test_set = self.args.test_datasets
        else:
            train_set = [self.args.dataset]
            validation_set = [self.args.dataset]
            test_set = [self.args.dataset]

        return train_set, validation_set, test_set

    """
    Command line parser
    """

    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset", choices=["meta-dataset", "ilsvrc_2012", "omniglot", "aircraft", "cu_birds",
                                                  "dtd", "quickdraw", "fungi", "vgg_flower", "traffic_sign", "mscoco",
                                                  "mnist", "cifar10", "cifar100", "from_file"], default="meta-dataset",
                            help="Dataset to use.")
        parser.add_argument('--test_datasets', nargs='+', help='Datasets to use for testing',
                            default=["quickdraw", "ilsvrc_2012", "omniglot", "aircraft", "cu_birds", "dtd",     "fungi",
                                     "vgg_flower", "traffic_sign", "mscoco", "mnist", "cifar10", "cifar100"])
        parser.add_argument("--data_path", default="../datasets", help="Path to dataset records.")
        parser.add_argument("--classifier", choices=["versa", "proto-nets", "mahalanobis", "mlpip"],
                            default="versa", help="Which classifier method to use.")
        parser.add_argument("--pretrained_resnet_path", default="learners/cnaps/models/pretrained_resnet.pt.tar",
                            help="Path to pretrained feature extractor model.")
        parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test",
                            help="Whether to run training only, testing only, or both training and testing.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=5e-4, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=16,
                            help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default='../checkpoints', help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--feature_adaptation", choices=["no_adaptation", "film", "film+ar"], default="film",
                            help="Method to adapt feature extractor parameters.")
        parser.add_argument("--batch_normalization", choices=["basic", "task_norm-i"],
                            default="basic", help="Normalization layer to use.")
        parser.add_argument("--training_iterations", "-i", type=int, default=110000,
                            help="Number of meta-training iterations.")
        parser.add_argument("--val_freq", type=int, default=10000, help="Number of iterations between validations.")
        parser.add_argument("--max_way_train", type=int, default=40,
                            help="Maximum way of meta-dataset meta-train task.")
        parser.add_argument("--max_way_test", type=int, default=50, help="Maximum way of meta-dataset meta-test task.")
        parser.add_argument("--max_support_train", type=int, default=400,
                            help="Maximum support set size of meta-dataset meta-train task.")
        parser.add_argument("--max_support_test", type=int, default=400,
                            help="Maximum support set size of meta-dataset meta-test task.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False,
                            action="store_true", help="Restart from latest checkpoint.")
        parser.add_argument("--way", type=int, default=5, help="Way of single dataset task.")
        parser.add_argument("--shot", type=int, default=1, help="Shots per class for context of single dataset task.")
        parser.add_argument("--query_train", type=int, default=10,
                            help="Shots per class for target  of single dataset task.")
        parser.add_argument("--query_test", type=int, default=10,
                            help="Shots per class for target  of single dataset task.")
        #parser.add_argument("--save_task", default=False,
        #                    help="Save all the tasks images to a pickle file. ")
        parser.add_argument("--do_not_freeze_feature_extractor", dest="do_not_freeze_feature_extractor", default=False,
                            action="store_true", help="If True, don't freeze the feature extractor.")
        parser.add_argument("--construct_coreset", default=False)
        parser.add_argument("--num_reruns", type=int, default=10,
                            help="Number of times to rerun the covering task when doing coreset construction")
        args = parser.parse_args()

        return args

    def run(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session:
            if self.args.mode == 'train' or self.args.mode == 'train_test':
                train_accuracies = []
                losses = []
                total_iterations = self.args.training_iterations
                for iteration in range(self.start_iteration, total_iterations):
                    torch.set_grad_enabled(True)
                    task_dict = self.dataset.get_train_task(session)
                    task_loss, task_accuracy = self.train_task(task_dict)
                    train_accuracies.append(task_accuracy)
                    losses.append(task_loss)

                    # optimize
                    if ((iteration + 1) % self.args.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    if (iteration + 1) % PRINT_FREQUENCY == 0:
                        # print training stats
                        print_and_log(self.logfile, 'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
                                      .format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
                                              torch.Tensor(train_accuracies).mean().item()))
                        train_accuracies = []
                        losses = []

                    if ((iteration + 1) % self.args.val_freq == 0) and (iteration + 1) != total_iterations:
                        # validate
                        accuracy_dict = self.validate(session)
                        self.validation_accuracies.print(self.logfile, accuracy_dict)
                        # save the model if validation is the best so far
                        if self.validation_accuracies.is_better(accuracy_dict):
                            self.validation_accuracies.replace(accuracy_dict)
                            torch.save(self.model.state_dict(), self.checkpoint_path_validation)
                            print_and_log(self.logfile, 'Best validation model was updated.')
                            print_and_log(self.logfile, '')
                        self.save_checkpoint(iteration + 1)

                # save the final model
                torch.save(self.model.state_dict(), self.checkpoint_path_final)

            if self.args.mode == 'train_test':
                self.test(self.checkpoint_path_final, session)
                self.test(self.checkpoint_path_validation, session)

            if self.args.mode == 'test':
                self.construct_coreset(self.args.test_model_path, session)
                #self.test_leave_one_out(self.args.test_model_path, session)

            self.logfile.close()

    def train_task(self, task_dict):
        context_images, target_images, context_labels, target_labels, _ = self.prepare_task(task_dict)

        target_logits = self.model(context_images, context_labels, target_images)
        task_loss = self.loss(target_logits, target_labels, self.device) / self.args.tasks_per_batch
        if self.args.feature_adaptation == 'film' or self.args.feature_adaptation == 'film+ar':
            if self.use_two_gpus():
                regularization_term = (self.model.feature_adaptation_network.regularization_term()).cuda(0)
            else:
                regularization_term = (self.model.feature_adaptation_network.regularization_term())
            regularizer_scaling = 0.001
            task_loss += regularizer_scaling * regularization_term
        task_accuracy = self.accuracy_fn(target_logits, target_labels)

        task_loss.backward(retain_graph=False)

        return task_loss, task_accuracy

    def validate(self, session):
        with torch.no_grad():
            accuracy_dict = {}
            for item in self.validation_set:
                accuracies = []
                for _ in range(NUM_VALIDATION_TASKS):
                    task_dict = self.dataset.get_validation_task(item, session)
                    context_images, target_images, context_labels, target_labels, _ = self.prepare_task(task_dict)
                    target_logits = self.model(context_images, context_labels, target_images)
                    accuracy = self.accuracy_fn(target_logits, target_labels)
                    accuracies.append(accuracy.item())
                    del target_logits

                accuracy = np.array(accuracies).mean() * 100.0
                confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

                accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence}

        return accuracy_dict

    def construct_coreset(self, path, session):
        print_and_log(self.logfile, "")  # add a blank line
        print_and_log(self.logfile, 'Constructing coreset with model {0:}: '.format(path))
        self.model = self.init_model()
        self.model.load_state_dict(torch.load(path))
        cifar_dataset = self.dataset 
        importance_scores = np.zeros((len(cifar_dataset.train_set), self.args.num_reruns))
        
        with torch.no_grad():
            for r in range(self.args.num_reruns):
                tasks = cifar_dataset.get_covering_tasks() # target shot
                import pdb; pdb.set_trace()
                ave_task_acc = 0.0
                for task in tqdm(tasks):
                    context_images, target_images, context_labels, target_labels = self.prepare_task(task, shuffle=False)
                    
                    logits = self.model(context_images, context_labels, target_images)
                    task_accuracy = self.accuracy_fn(logits, target_labels).item()
                    ave_task_acc += task_accuracy
                    loo_accs = self.leave_one_out_accuracies(context_images, context_labels, target_images, target_labels)
                    scores = task_accuracy - loo_accs
                    for i, index in enumerate(task['context_indices']):
                        assert abs(importance_scores[index][r] - 0) < 1.0e-15
                        importance_scores[index][r] = scores[i]
                print_and_log(self.logfile, "Rerun {}, ave task accuracy: {:2.4f}".format(r, ave_task_acc/float(len(tasks))))
            for i in range(len(cifar_dataset.train_set)):
                if abs(importance_scores[i].mean()) > 1.0e-10:
                    print_and_log(self.logfile, '{}: {:2.4f}+/-{:2.4f}'.format(i, importance_scores[i].mean(), importance_scores[i].std()))
        np.save(os.path.join(self.args.checkpoint_dir, 'importance_scores.npy'), importance_scores)
        import pdb; pdb.set_trace()


    def leave_one_out_accuracies(self, context_images, context_labels, target_images, target_labels):
        accuracy_loo = np.ndarray((len(context_images)))
        for i in range(context_images.shape[0]):
            if i == 0:
                context_images_loo = context_images[i + 1:]
                context_labels_loo = context_labels[i + 1:]
            else:
                context_images_loo = torch.cat((context_images[0:i], context_images[i + 1:]), 0)
                context_labels_loo = torch.cat((context_labels[0:i], context_labels[i + 1:]), 0)

            logits = self.model(context_images_loo, context_labels_loo, target_images)
            accuracy = self.accuracy_fn(logits, target_labels).item()
            accuracy_loo[i] = accuracy
        return accuracy_loo

    def test_leave_one_out(self, path, session):
        print_and_log(self.logfile, "")  # add a blank line
        print_and_log(self.logfile, 'Testing model {0:}: '.format(path))
        self.model = self.init_model()
        self.model.load_state_dict(torch.load(path))

        with torch.no_grad():
            task_dict = self.dataset.get_test_task(0, session) # target shot
            context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict, shuffle=False)

            logits = self.model(context_images, context_labels, target_images)
            true_accuracy = self.accuracy_fn(logits, target_labels).item()
            print_and_log(self.logfile, 'True accuracy: {0:3.5f}'.format(true_accuracy))
            print_and_log(self.logfile, 'Class Labels: {}'.format(context_labels))

            accuracy_loo = []
            for i in range(context_images.shape[0]):
                if i == 0:
                    context_images_loo = context_images[i + 1:]
                    context_labels_loo = context_labels[i + 1:]
                else:
                    context_images_loo = torch.cat((context_images[0:i], context_images[i + 1:]), 0)
                    context_labels_loo = torch.cat((context_labels[0:i], context_labels[i + 1:]), 0)

                logits = self.model(context_images_loo, context_labels_loo, target_images)
                accuracy = self.accuracy_fn(logits, target_labels).item()

                print_and_log(self.logfile, 'Loo {0:} accuracy: {1:3.5f}'.format(i, true_accuracy - accuracy))
                accuracy_loo.append(accuracy)
                for c in range(0, self.args.way):
                    target_images_c = target_images[c*(self.args.query_test): (c+1)*self.args.query_test]
                    target_labels_c = target_labels[c*(self.args.query_test): (c+1)*self.args.query_test]
                    logits = self.model(context_images_loo, context_labels_loo, target_images_c)
                    accuracy = self.accuracy_fn(logits, target_labels_c).item()
                    print_and_log(self.logfile, '\tLoo {0:} class {1:}  accuracy: {2:3.5f}'.format(i, c, true_accuracy - accuracy))
                del logits



    def test(self, path, session):
        print_and_log(self.logfile, "")  # add a blank line
        print_and_log(self.logfile, 'Testing model {0:}: '.format(path))
        self.model = self.init_model()
        self.model.load_state_dict(torch.load(path))

        with torch.no_grad():
            for item in self.test_set:
                accuracies = []
                for _ in range(NUM_TEST_TASKS):
                    task_dict = self.dataset.get_test_task(item, session)
                    context_images, target_images, context_labels, target_labels, _ = self.prepare_task(task_dict)
                    target_logits = self.model(context_images, context_labels, target_images)
                    accuracy = self.accuracy_fn(target_logits, target_labels)
                    accuracies.append(accuracy.item())
                    del target_logits

                accuracy = np.array(accuracies).mean() * 100.0
                accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

                print_and_log(self.logfile, '{0:}: {1:3.1f}+/-{2:2.1f}'.format(item, accuracy, accuracy_confidence))

    def calc_accuracy(self, context_images, context_labels, target_images, target_labels):
        logits = self.model(context_images, context_labels, target_images)
        acc = torch.mean(torch.eq(target_labels.long(), torch.argmax(logits, dim=-1).long()).float()).item()
        del logits
        return acc

    def print_average_accuracy(self, accuracies, descriptor, item):
        write_to_log(self.logfile, '{}, {}'.format(descriptor, item))
        write_to_log(self.debugfile,'{}'.format(accuracies))
        accuracy = np.array(accuracies).mean() * 100.0
        accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
        print_and_log(self.logfile,
                      '{0:} {1:}: {2:3.1f}+/-{3:2.1f}'.format(descriptor, item, accuracy, accuracy_confidence))

    def save_image_pair(self, adv_img, clean_img, task_no, index):
        utils.save_image(adv_img.cpu().detach().numpy(),os.path.join(self.checkpoint_dir, 'adv_task_{}_index_{}.png'.format(task_no, index)))
        utils.save_image(clean_img, os.path.join(self.checkpoint_dir, 'in_task_{}_index_{}.png'.format(task_no, index)))

    def prepare_task(self, task_dict, shuffle=True):
        context_images_np, context_labels_np = task_dict['context_images'], task_dict['context_labels']
        target_images_np, target_labels_np = task_dict['target_images'], task_dict['target_labels']

        context_images_np = context_images_np.transpose([0, 3, 1, 2])

        if shuffle:
            context_images_np, context_labels_np = self.shuffle(context_images_np, context_labels_np)
        context_images = torch.from_numpy(context_images_np)
        context_labels = torch.from_numpy(context_labels_np)

        target_images_np = target_images_np.transpose([0, 3, 1, 2])
        if shuffle:
            target_images_np, target_labels_np = self.shuffle(target_images_np, target_labels_np)
        target_images = torch.from_numpy(target_images_np)
        target_labels = torch.from_numpy(target_labels_np)
        target_labels = target_labels.type(torch.LongTensor)

        context_images = context_images.to(self.device)
        target_images = target_images.to(self.device)
        target_labels = target_labels.to(self.device)
        context_labels = context_labels.to(self.device)

        return context_images, target_images, context_labels, target_labels

    def loss_fn(self, test_logits_sample, test_labels):
        """
        Compute the classification loss.
        """
        size = test_logits_sample.size()
        sample_count = size[0]  # scalar for the loop counter
        num_samples = torch.tensor([sample_count], dtype=torch.float, device=self.device, requires_grad=False)

        log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=self.device)
        for sample in range(sample_count):
            log_py[sample] = -F.cross_entropy(test_logits_sample[sample], test_labels, reduction='none')
        score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)
        return -torch.sum(score, dim=0)

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]

    def use_two_gpus(self):
        use_two_gpus = False
        if self.args.dataset == "meta-dataset":
            if self.args.feature_adaptation == "film+ar" or \
                    self.args.batch_normalization == "task_norm-i":
                use_two_gpus = True  # These models do not fit on one GPU, so use model parallelism.

        return use_two_gpus

    def save_checkpoint(self, iteration):
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.validation_accuracies.get_current_best_accuracy_dict(),
        }, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'checkpoint.pt'))
        self.start_iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.validation_accuracies.replace(checkpoint['best_accuracy'])

    def register_extra_parameters(self, model):
        for module in model.modules():
            if isinstance(module, TaskNormI):
                module.register_extra_weights()


if __name__ == "__main__":
    main()
