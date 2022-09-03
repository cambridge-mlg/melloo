import torch
import torchvision.transforms.functional as tvf
from torch.distributions import Categorical
import numpy as np
from numpy.random import default_rng
import argparse
import os
from tqdm import tqdm
from utils import MetaLearningState
import utils
from model import FewShotClassifier
from normalization_layers import TaskNormI
from dataset import get_dataset_reader
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import math
import matplotlib
import matplotlib.pyplot as plt
import gc
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import sklearn.metrics
import metaloo
import metrics

from scipy.stats import kendalltau
from sklearn.metrics.pairwise import rbf_kernel

NUM_VALIDATION_TASKS = 200
PRINT_FREQUENCY = 1000
rng = default_rng()


def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()
        self.log_files = utils.LogFiles(self.args.checkpoint_dir, self.args.resume_from_checkpoint, self.args.mode == "test")
        self.logger = utils.Logger(self.args.checkpoint_dir, "log.txt")

        self.logger.print_and_log("Options: %s\n" % self.args)
        self.logger.print_and_log("Checkpoint Directory: %s\n" % self.log_files.checkpoint_dir)
        
        self.metrics = new metrics.Metrics(self.args.checkpoint_dir, self.args.way, self.args.top_k, self.logger)
        self.representer_args = {
                "l2_regularize_classifier": self.args.l2_regularize_classifier,
                "l2_lambda": self.args.l2_lambda,
                "kernel_agg": self.args.kernel_agg,
            }

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.train_set, self.validation_set, self.test_set = self.init_data()
        value_tracking = self.args.task_type == "generate_coreset_discard"
        
        self.dataset = get_dataset_reader(
            args=self.args,
            train_set=self.train_set,
            validation_set=self.validation_set,
            test_set=self.test_set,
            device=self.device,
            value_tracking=value_tracking)

        self.loss = utils.cross_entropy_loss
        self.accuracy_fn = utils.categorical_accuracy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.validation_accuracies = utils.ValidationAccuracies(self.validation_set)
        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()
        # Not sure whether to use shot or max_support_test
        if self.args.task_type == "shot_selection":
            if self.args.spread_constraint == "by_class":
                assert self.args.top_k <= self.args.shot
            else :
                assert self.args.top_k <= self.args.shot * self.args.way
            assert self.args.selection_mode != "drop"
        elif self.args.task_type == "generate_coreset":
            assert self.args.selection_mode != "drop"
        else:
            assert self.args.spread_constraint == "none"
            if self.args.task_type == "noisy_shots" or self.args.task_type == "generate_coreset_discard":
                assert self.args.selection_mode == "drop"
            else:
                assert self.args.selection_mode != "drop"
                
        self.top_k = self.args.top_k

    def init_model(self):
        use_two_gpus = self.use_two_gpus()
        model = FewShotClassifier(device=self.device, use_two_gpus=use_two_gpus, args=self.args).to(self.device)
        self.register_extra_parameters(model)

        # set encoder is always in train mode (it only sees context data).
        # Feature extractor gets switched in model depending on args.batch_normalization.
        model.train()
        if use_two_gpus:
            model.distribute_model()
        return model

    def init_data(self):
        if self.args.dataset == "meta-dataset":
            train_set = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower']
            validation_set = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi',
                              'vgg_flower', 'mscoco']
            test_set = self.args.test_datasets
        elif self.args.dataset == "meta-dataset_ilsvrc_only":
            train_set = ['ilsvrc_2012']
            validation_set = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi',
                              'vgg_flower', 'mscoco']
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

        parser.add_argument("--dataset", choices=["meta-dataset", "meta-dataset_ilsvrc_only", "ilsvrc_2012", "omniglot", "aircraft", "cu_birds",
                                                  "dtd", "quickdraw", "fungi", "vgg_flower", "traffic_sign", "mscoco",
                                                  "mnist", "cifar10", "cifar100", "split-cifar10", "split-mnist"], default="meta-dataset",
                            help="Dataset to use.")
        parser.add_argument("--dataset_reader", choices=["official", "pytorch"], default="official",
                            help="Dataset reader to use.")
        parser.add_argument("--classifier", choices=["protonets_euclidean",
                                                     "protonets_attention",
                                                     "protonets_cross_transformer",
                                                     "protonets_mahalanobis"],
                            default="versa", help="Which classifier method to use.")
        parser.add_argument('--test_datasets', nargs='+', help='Datasets to use for testing',
                            default=["ilsvrc_2012", "omniglot", "aircraft", "cu_birds", "dtd", "quickdraw", "fungi",
                                     "vgg_flower", "traffic_sign", "mscoco", "mnist", "cifar10", "cifar100"])
        parser.add_argument("--data_path", default="../datasets", help="Path to dataset records.")
        parser.add_argument("--pretrained_resnet_path", default="../models/pretrained_resnet.pt.tar",
                            help="Path to pretrained feature extractor model.")
        parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test",
                            help="Whether to run training only, testing only, or both training and testing.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=16,
                            help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default='../checkpoints', help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--feature_adaptation", choices=["no_adaptation", "film"], default="no_adaptation",
                            help="Method to adapt feature extractor parameters.")
        parser.add_argument("--batch_normalization", choices=["basic", "standard", "task_norm-i"],
                            default="basic", help="Normalization layer to use.")
        parser.add_argument("--training_iterations", "-i", type=int, default=60000, help="Number of meta-training iterations.")
        parser.add_argument("--val_freq", type=int, default=5000, help="Number of iterations between validations.")
        parser.add_argument("--max_way_train", type=int, default=50, help="Maximum way of meta-train task.")
        parser.add_argument("--max_way_test", type=int, default=50, help="Maximum way of meta-test task.")
        parser.add_argument("--max_support_train", type=int, default=50000, help="Maximum support set size of meta-train task.")
        parser.add_argument("--max_support_test", type=int, default=50000, help="Maximum support set size of meta-test task.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False, action="store_true",
                            help="Restart from ltest checkpoint.")
        parser.add_argument("--way", type=int, default=5, help="Way of meta-train task.")
        parser.add_argument("--shot", type=int, default=1, help="Shots per class for context.")
        parser.add_argument("--query_train", type=int, default=10, help="Shots per class for target during meta-training.")
        parser.add_argument("--query_test", type=int, default=10, help="Shots per class for target during meta-testing.")
        parser.add_argument("--image_size", type=int, default=84, help="Image height and width.")
        parser.add_argument("--do_not_freeze_feature_extractor", dest="do_not_freeze_feature_extractor", default=False,
                            action="store_true", help="If True, don't freeze the feature extractor.")
        parser.add_argument("--compute_multi_query_accuracy", dest="compute_multi_query_accuracy", default=False,
                            action="store_true", help="If True, compute joint accuracy of all the target instances of a class using varios methods.")
        parser.add_argument("--num_attention_heads", type=int, default=8, help="Number of heads in multi-head attention.")
        parser.add_argument("--attention_temperature", type=float, default=1.0,
                            help="Temperature used in dot-product attention softmax.")
        # Regularization options when training
        parser.add_argument("--l2_lambda", type=float, default=0.0001,
                            help="Value of lambda when doing L2 regularization on the classifier head")
        parser.add_argument("--l2_regularize_classifier", dest="l2_regularize_classifier", default=False,
                            action="store_true", help="If True, perform l2 regularization on the classifier head.")
        # Melloo task type and related settings
        parser.add_argument("--task_type", choices=["generate_coreset", "generate_coreset_discard", "noisy_shots", "shot_selection"], default="generate_coreset",
                            help="Task to perform. Options are generate_coreset, noisy_shots or shot_selection")
        parser.add_argument("--tasks", type=int, default=10, help="Number of test tasks to do")
        parser.add_argument("--test_case", choices=["default", "bimodal", "noise", "unrelated"], default="default",
                            help="Specifiy a specific test case to try. Currently only default and bimodal are implemented")
        parser.add_argument("--top_k", type=int, default=2, help="How many points to retain from each class")
        parser.add_argument("--selection_mode", choices=["top_k", "multinomial", "divine", "drop"], default="top_k",
                            help="How to select the candidates from the importance weights. Drop option only works for noisy shot selection at present.")
        parser.add_argument("--importance_mode", choices=["attention", "loo", "representer", "random", "all"], default="all",
                            help="How to calculate candidate importance")
        parser.add_argument("--kernel_agg", choices=["sum", "sum_abs", "class"], default="class",
                            help="When using representer importance, how to aggregate kernel information")
        # Shot selection settings
        parser.add_argument("--spread_constraint", choices=["by_class", "none"], default="by_class",
                            help="Spread coresets over classes (by_class) or no constraint (none)")
        # Noisy task settings
        parser.add_argument("--drop_rate", type=float, default=0.1, help="Percentage of points to drop (as float in [0,1])")
        parser.add_argument("--noise_type", choices=["mislabel", "ood"], default="mislabel",
                            help="Type of noise to use for noisy shot detection")
        parser.add_argument("--DEBUG_bimodal_code", choices=["old", "funky"], default="funky")
        parser.add_argument("--error_rate", type=float, default=0.1,
                            help="Rate at which noise is introduced. For mislabelling, this is the percentage of the context set to mislabel. For ood, this how many ood patterns are shuffled into the other classes (calculated as num context patterns * error rate)")
        
        args = parser.parse_args()
        return args

    def run(self):
        if self.args.mode == 'train' or self.args.mode == 'train_test':
            train_accuracies = []
            losses = []
            total_iterations = self.args.training_iterations
            for iteration in tqdm(range(self.start_iteration, total_iterations), dynamic_ncols=True):
                torch.set_grad_enabled(True)
                task_dict = self.dataset.get_train_task()
                task_loss, task_accuracy = self.train_task(task_dict)
                train_accuracies.append(task_accuracy)
                losses.append(task_loss)

                # optimize
                if ((iteration + 1) % self.args.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if (iteration + 1) % PRINT_FREQUENCY == 0:
                    # print training stats
                    self.logger.print_and_log(
                        'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}, Learning Rate: {:.7f}'
                            .format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
                                    torch.Tensor(train_accuracies).mean().item(), self.optimizer.param_groups[0]['lr']))
                    # print("Temperature = {}".format(1.0 / self.model.cross_attention.scale.item()))
                    train_accuracies = []
                    losses = []
                    self.save_checkpoint(iteration + 1)

                if ((iteration + 1) % self.args.val_freq == 0) and (iteration + 1) != total_iterations:
                    # validate
                    accuracy_dict = self.validate()
                    self.validation_accuracies.print(self.logger, accuracy_dict)
                    # save the model if validation is the best so far
                    if self.validation_accuracies.is_better(accuracy_dict):
                        self.validation_accuracies.replace(accuracy_dict)
                        torch.save(self.model.state_dict(), self.log_files.best_validation_model_path)
                        self.logger.print_and_log('Best validation model was updated.')
                        self.logger.print_and_log('')
                    torch.save(self.model.state_dict(), os.path.join(self.log_files.checkpoint_dir,
                                                                     "model_{}.pt".format(iteration + 1)))

            # save the final model
            torch.save(self.model.state_dict(), self.log_files.fully_trained_model_path)

        if self.args.mode == 'train_test':
            self.test(self.log_files.fully_trained_model_path)
            self.test(self.log_files.best_validation_model_path)
        if self.args.mode == 'test':
            if self.args.task_type == "generate_coreset":
                self.generate_coreset(self.args.test_model_path)
            elif self.args.task_type == "generate_coreset_discard":
                self.generate_coreset_discard(self.args.test_model_path)
                #self.save_coreset_from_ranking(self.args.test_model_path)
            elif self.args.task_type == "noisy_shots":
                self.detect_noisy_shots(self.args.test_model_path)
            elif self.args.task_type == "shot_selection":

                if self.args.test_case == "bimodal":
                    if self.args.DEBUG_bimodal_code == "funky":
                        self.funky_bimodal_task(self.args.test_model_path)
                    else:
                        self.select_shots(self.args.test_model_path)
                else:
                    self.select_shots(self.args.test_model_path)
            else:
                print("Unsupported task specified")

    def get_from_gpu(self, value):
        if self.use_two_gpus():
            return value.cuda(0)
        else:
            return value

    def train_task(self, task_dict):
        context_images, target_images, context_labels, target_labels = utils.prepare_task(task_dict, self.device)

        target_logits = self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TRAIN)
        task_loss = self.loss(target_logits, target_labels)
        if self.args.feature_adaptation == 'film':
            regularization_term = self.get_from_gpu(self.model.feature_adaptation_network.regularization_term())
            regularizer_scaling = 0.001
            task_loss += regularizer_scaling * regularization_term
            
        if self.args.l2_regularize_classifier:
            regularization_term = self.get_from_gpu(self.model.classifer_regularization_term())
            task_loss += self.args.l2_lambda * regularization_term
        task_accuracy = self.accuracy_fn(target_logits, target_labels)

        task_loss.backward(retain_graph=False)

        return task_loss, task_accuracy

    def validate(self):
        with torch.no_grad():
            accuracy_dict ={}
            for item in self.validation_set:
                accuracies = []
                for _ in range(NUM_VALIDATION_TASKS):
                    task_dict = self.dataset.get_validation_task(item)
                    context_images, target_images, context_labels, target_labels = utils.prepare_task(task_dict, self.device)
                    target_logits = self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
                    accuracy = self.accuracy_fn(target_logits, target_labels)
                    accuracies.append(accuracy.item())
                    del target_logits

                accuracy = np.array(accuracies).mean() * 100.0
                confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

                accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence}

        return accuracy_dict

    def test(self, path):
        self.logger.print_and_log("")  # add a blank line
        self.logger.print_and_log('Testing model {0:}: '.format(path))
        self.model = self.init_model()
        if path != 'None':
            self.model.load_state_dict(torch.load(path))

        for item in self.test_set:
            accuracies = []
            protonets_accuracies = []
            for _ in range(self.args.tasks):
                task_dict = self.dataset.get_test_task(item)
                context_images, target_images, context_labels, target_labels = utils.prepare_task(task_dict, self.device)
                # Do the forward pass with the target set so that we can get the feature embeddings
                with torch.no_grad():
                    target_logits = self.model(context_images, context_labels, target_images, target_labels,
                                               MetaLearningState.META_TEST)
                                               
                # Make a copy of the attention weights and the target features, otherwise we get a reference to what the model is storing
                # (and we're about to send another task through it, so that's not what we want)
                context_features, target_features, attention_weights = self.model.context_features, self.model.target_features.clone(), self.model.attention_weights
                weights_per_query_point = self.reshape_attention_weights(attention_weights, context_labels, len(target_labels)).cpu()
                
                # Now do the forward pass with the context set for the representer points:
                context_logits = self.model(context_images, context_labels, context_images, context_labels,
                                           MetaLearningState.META_TEST)
                task_loss = self.loss(context_logits, context_labels)
                regularization_term = (self.model.feature_adaptation_network.regularization_term())
                regularizer_scaling = 0.001
                task_loss += regularizer_scaling * regularization_term

                if self.args.l2_regularize_classifier:
                    classifier_regularization_term = self.model.classifer_regularization_term()
                    task_loss += self.args.l2_lambda * classifier_regularization_term
                    
                # Representer values calculation    
                dl_dphi = torch.autograd.grad(task_loss, context_logits, retain_graph=True)[0]
                alphas = dl_dphi/(-2.0 * self.args.l2_lambda * float(len(context_labels)))
                alphas_t = alphas.transpose(1,0)
                
                feature_prod = torch.matmul(context_features, target_features.transpose(1,0))

                kernels_by_class = []
                kernels_sum_abs = []
                kernels_sum = []
                for k in range(alphas.shape[1]):
                    alphas_k = alphas_t[k]
                    tmp_mat = alphas_k.unsqueeze(1).expand(len(context_labels), len(target_labels))
                    kernels_k = tmp_mat * feature_prod
                    kernels_by_class.append(kernels_k.unsqueeze(0))

                representers = torch.cat(kernels_by_class, dim=0)
                
                representer_tmp = []
                for c in torch.unique(context_labels):
                    c_indices = utils.extract_class_indices(context_labels, c)
                    for i in c_indices:    
                        representer_tmp.append(representers[c][i])
                representer_per_query_point = torch.stack(representer_tmp).transpose(1,0)
                representer_approx = torch.matmul(alphas.transpose(1,0), torch.matmul(context_features, target_features.transpose(1,0)))
                representer_per_query_point = representer_per_query_point.cpu()
                
                representer_summed = representers.sum(dim=0).transpose(1,0).cpu()
                representer_abs_sum = representers.abs().sum(dim=0).transpose(1,0).cpu()
                
                plt.scatter(weights_per_query_point, representer_per_query_point)
                plt.xlabel("attention weights")
                plt.ylabel("representer weights")
                plt.savefig(os.path.join(self.args.checkpoint_dir, "attention_vs_representer.png"))
                plt.close()

                plt.scatter(weights_per_query_point, representer_abs_sum)
                plt.xlabel("attention weights")
                plt.ylabel("representer weights (sum abs)")
                plt.savefig(os.path.join(self.args.checkpoint_dir, "attention_vs_representer_sum_abs.png"))
                plt.close()

                plt.scatter(weights_per_query_point, representer_summed)
                plt.xlabel("attention weights")
                plt.ylabel("representer weights (sum)")
                plt.savefig(os.path.join(self.args.checkpoint_dir, "attention_vs_representer_summed.png"))
                plt.close()

                ave_corr, ave_num_intersected = self.metrics.compare_rankings(weights_per_query_point, representer_per_query_point, "Attention vs Representer", weights=True)
                #ave_corr_mag, ave_num_intersected_mag = self.metrics.compare_rankings(weights_per_query_point, representer_per_query_point, "Attention vs Representer", weights=True)
                ave_corr_s, ave_num_intersected_s = self.metrics.compare_rankings(weights_per_query_point, representer_summed, "Attention vs Representer Summed", weights=True)
                #ave_corr_mag_s, ave_num_intersected_mag_s = self.metrics.compare_rankings(weights_per_query_point, representer_summed, "Attention vs Representer Summed", weights=True)
                ave_corr_abs_s, ave_num_intersected_abs_s = self.metrics.compare_rankings(weights_per_query_point, representer_abs_sum, "Attention vs Representer Sum Abs", weights=True)
                #ave_corr_mag_abs_s, ave_num_intersected_mag_abs_s = self.metrics.compare_rankings(weights_per_query_point, representer_abs_sum, "Attention vs Representer Sum Abs", weights=True)
                
            del target_logits

            accuracy = np.array(accuracies).mean() * 100.0
            accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

            self.logger.print_and_log('{0:}: {1:3.1f}+/-{2:2.1f}'.format(item, accuracy, accuracy_confidence))
            

    def random_weights(self, context_images, context_labels, target_images, target_labels):
        return torch.tensor(np.random.rand(len(target_labels), len(context_labels)))

    

    def save_coreset_from_ranking(self, path):
        self.logger.print_and_log("")  # add a blank line
        self.logger.print_and_log("Saving out top candidates from ranking")  # add a blank line
        
        
        image_rankings = pickle.load(open(os.path.join(self.args.checkpoint_dir, "rankings.pickle"), "rb"))

        for key in image_rankings[0].keys():
            weights, image_ids = metaloo.weights_from_multirankings(image_rankings, key)
            candidate_indices = self.select_top_k(self.args.top_k, weights, self.args.spread_constraint)
            candidate_ids = image_ids[candidate_indices]

            task_dict = self.dataset.get_task_from_ids(candidate_ids)
            context_images, _, _, _ = utils.prepare_task(task_dict, self.device)

            self.metrics.save_image_set(0, context_images, "selected_{}".format(key))

    def generate_coreset_discard(self, path):
        #import pdb; pdb.set_trace()
        self.logger.print_and_log("")  # add a blank line
        self.logger.print_and_log('Generating coreset (by discard) using model {0:}: '.format(path))
        self.model = self.init_model()
        if path != 'None':
            self.model.load_state_dict(torch.load(path))
        ranking_mode = self.args.importance_mode  
        save_out_interval = 500
        if self.args.tasks <= 1000:
            save_out_interval = 100
            '''
            if self.args.top_k == 1:
                save_out_interval = 1
            else:
                save_out_interval = 100
            '''

        for item in self.test_set:
            accuracies = []
            losses = []
            entropies = []
            restart_checkpoints = []
            if self.args.importance_mode == 'all':
                self.metrics.plot_scatter("Error - coreset construction by discard does not support doing all importance modes simultaneously")
                return  
            # Start with the original, random context set
            task_dict = self.dataset.get_test_task(item)
            context_images, target_images, context_labels, target_labels = utils.prepare_task(task_dict, self.device)
            image_ids = task_dict["context_ids"]
            no_drop_count = 0
            for ti in tqdm(range(self.args.tasks), dynamic_ncols=True):
                
                with torch.no_grad():
                    target_logits = self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
                    task_accuracy = self.accuracy_fn(target_logits, target_labels).item()
                    task_loss = self.loss(target_logits, target_labels, reduce=True).item()
                    accuracies.append(task_accuracy)
                    losses.append(task_loss)
                    if ti % save_out_interval == 0 or ti == self.args.tasks-1:
                        self.metrics.plot_confusion_matrix(target_labels, "confusion_{}.png".format(ti), logits=target_logits)
                    del target_logits

                one_hot = torch.nn.functional.one_hot(context_labels).to("cpu")
                probs = one_hot.sum(dim=0)/float(len(context_labels))
                entropies.append(Categorical(probs=probs).entropy())
                
                # Save the target/context features
                # We could optimize by only doing this if we're doing attention or divine selection, but for now whatever, it's a single forward pass
                if ranking_mode == "attention":
                    with torch.no_grad():
                        self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
                        # Make a copy of the target features, otherwise we get a reference to what the model is storing
                        # (and we're about to send another task through it, so that's not what we want)
                        context_features, target_features = self.model.context_features.clone(), self.model.target_features.clone()

                if ranking_mode == 'loo':
                    weight_per_qp = metaloo.calculate_loo(self.model, self.loss, context_images, context_labels, target_images, target_labels)
                elif ranking_mode == 'attention':
                    # Make a copy of the attention weights otherwise we get a reference to what the model is storing
                    with torch.no_grad():
                        self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
                    attention_weights = self.model.attention_weights.copy()
                    weight_per_qp = self.reshape_attention_weights(attention_weights, context_labels, len(target_labels)).cpu()
                elif ranking_mode == 'representer':
                    weight_per_qp = metaloo.calculate_representer_values(self.model, self.loss, context_images, context_labels, context_features, target_labels, target_features, self.representer_args, self.representer_args)
                elif ranking_mode == 'random':
                    weight_per_qp = self.random_weights(context_images, context_labels, target_images, target_labels)
                    
                # May want to normalize here:
                weights = weight_per_qp.sum(dim=0)
                rankings = torch.argsort(weights, descending=True)
                
                # Discard least valuable point/s
                candidate_indices, dropped_indices = metaloo.select_by_dropping(weights, number=self.args.top_k)
                candidate_ids = image_ids[candidate_indices]
                dropped_ids = image_ids[dropped_indices]
                if len(dropped_indices) == 1:
                    dropped_ids = np.array([dropped_ids])
                self.dataset.mark_discarded(dropped_ids)

                
                old_images = context_images[dropped_indices].clone()
                old_labels = context_labels[dropped_indices].clone()
                    
                ti_start = ti
                while ti < self.args.tasks - 1:
                    # Request new points to replace those
                    new_images, new_labels, new_ids = self.dataset.sample_new_context_points(self.args.top_k)
                    # Re-prepare data
                    new_images, new_labels = utils.move_set_to_cuda(new_images, new_labels, self.device)

                    for i, idd in enumerate(new_ids):
                        context_images[dropped_indices[i]] = new_images[i]
                        context_labels[dropped_indices[i]] = new_labels[i]
                        image_ids[dropped_indices[i]] = idd
                    if len(context_labels.unique()) != len(target_labels.unique()):
                        # We have dropped all points of a class; penalise heavily
                        new_loss = 10000
                    else:
                        # Try the new points; if it's worse, swap the old ones back in
                        new_logits = self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
                        #print("new_logits shape: {} target_labels shape: {} target_label classes {} context_label classes {}".format(new_logits.shape, target_labels.shape, target_labels.unique(), context_labels.unique()))
                        new_loss = self.loss(new_logits, target_labels, reduce=True).item()
                    #TODO: Do we want to compare accuracies or losses? I wouldn't thought losses...
                    if new_loss <= losses[-1]:
                        # New task improves or is equal to old accuracy; continue with loo 
                        no_drop_count = 0
                        break
                    else:
                        no_drop_count += 1
                        ti += 1
                        
                        if no_drop_count < 5:
                            self.dataset.mark_discarded(new_ids) # Try again
                        # If we've gone 5 or more iterations without dropping anything, do a random restart
                        if no_drop_count >= 5:
                            # Mark the "stuck" context set as discarded; has to happen before we reset the list of iamge_ids because the ValueTrackingDataset 
                            # knows what has been issued; there is a small chance that we might get reissued some of the same points
                            # import pdb; pdb.set_trace()
                            self.dataset.mark_discarded(image_ids)
                            print("Doing random restart")
                            #import pdb; pdb.set_trace()
                            # Swap the old ones back in so we get back to the set that first got stuck
                            for i, dropped_id in enumerate(dropped_ids):
                              image_ids[dropped_indices[i]] = dropped_id
                            # Save current context set
                            # Save this context set's loss
                            current_info = {"context_ids": np.copy(image_ids), "accuracy": accuracies[-1], "loss": losses[-1]}
                            # Request whole new context set; force reset so we don't get unbalanced classes
                            context_images, context_labels, image_ids = self.dataset.sample_new_context_points(context_images.shape[0], force_reset=True)
                            # reset drop count
                            no_drop_count = 0;
                            while len(context_labels.unique()) != len(target_labels.unique()):
                                print("Somehow whole new context set was missing a class, retrying")
                                # Request whole new context set; force reset so we don't get unbalanced classes
                                context_images, context_labels, image_ids = self.dataset.sample_new_context_points(context_images.shape[0], force_reset=True)
    
                            # Re-prepare data
                            context_images, context_labels = utils.move_set_to_cuda(context_images, context_labels, self.device)

                            restart_checkpoints.append(current_info)
                            break
                        

                if ti % save_out_interval == 0 or (ti - ti_start > save_out_interval):
                    self.metrics.save_image_set(ti, context_images[candidate_indices], "keep_{}".format(ti), labels=context_labels[candidate_indices])
                    self.metrics.save_image_set(ti, old_images, "discard_{}".format(ti), labels=old_labels)
                    self.metrics.save_image_set(ti, context_images[dropped_indices], "new_{}".format(ti), labels=context_labels[dropped_indices])
                    self.metrics.save_image_set(ti, target_images, "target_{}".format(ti), labels=target_labels)


                    # Remove the old points, replace with new points

                    # Get new query set as well
                    #if ti == int(self.args.tasks/2):
                    #    target_images, target_labels, _ = self.dataset.get_query_set()
                    #    target_images, target_labels = utils.move_set_to_cuda(target_images ,target_labels, self.device)

            eval_accuracies = []
            num_eval_tasks = 100
            eval_predictions = []
            eval_labels = []
            eval_losses = []
            for te in range(num_eval_tasks):
                with torch.no_grad():
                    target_images, target_labels, _ = self.dataset.get_query_set()
                    eval_labels.append(target_labels.cpu())
                    target_images, target_labels = utils.move_set_to_cuda(target_images ,target_labels, self.device)
                    target_logits = self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
                    eval_predictions.append(target_logits.argmax(axis=1).cpu())
                    eval_losses.append(self.loss(target_logits, target_labels, reduce=True).item())
                    task_accuracy = self.accuracy_fn(target_logits, target_labels).item()
                    eval_accuracies.append(task_accuracy)
                    del target_logits
            all_eval_labels = torch.cat(eval_labels)
            all_eval_predictions = torch.cat(eval_predictions)
            self.metrics.plot_confusion_matrix(all_eval_labels, pred_labels=all_eval_predictions, filename="confusion_eval.png")
            
            self.metrics.plot_scatter(accuracies, item, 'Accuracy')
            self.metrics.plot_scatter(eval_accuracies, item, 'Eval Accuracy')
            self.metrics.plot_scatter(eval_losses, item, 'Eval Loss')
            self.logger.print_and_log("Number of no_drop iterations: {}".format(no_drop_count))
            self.metrics.save_image_set(ti, context_images, "context_final".format(ti), labels=context_labels)
            self.metrics.plot_and_log(accuracies, "Accuracies over tasks", "accuracies.png")
            self.metrics.plot_and_log(losses, "Losses over tasks", "losses.png")
            self.metrics.plot_and_log(entropies, "Entropy of context labels", "entropy.png")
            self.metrics.bar_plot_and_log(list(self.dataset.returned_label_counts.keys()), self.dataset.returned_label_counts.values(), "Returned label counts: ", "returned_label_counts.png")
            self.metrics.plot_hist(context_labels.cpu(), np.arange(10), "final_context_distrib", title='Final Context Distribution', x_label='class', y_label='count', density=False)
            import pdb; pdb.set_trace()
            self.logger.log("{}".format(restart_checkpoints))

    def generate_coreset(self, path):
        self.logger.print_and_log("")  # add a blank line
        self.logger.print_and_log('Generating coreset (full scoring) using model {0:}: '.format(path))
        self.model = self.init_model()

        if path != 'None':
            self.model.load_state_dict(torch.load(path))

        for item in self.test_set:
            accuracies_full = []
            accuracies = {}
            if self.args.importance_mode == 'all':
                ranking_modes = ['loo', 'random'] #'attention', 'representer', 'random']
            else:
                ranking_modes = [self.args.importance_mode]
                
            # A dictionary mapping an image id to a list of its rankings
            image_rankings = {}
                
            for mode in ranking_modes:
                accuracies[mode] = []
                
            for ti in tqdm(range(self.args.tasks), dynamic_ncols=True):
                task_dict = self.dataset.get_test_task(item)
                context_images, target_images, context_labels, target_labels = utils.prepare_task(task_dict, self.device)

                #rankings_per_qp = {}
                weights_per_qp = {}
                #rankings = {}
                weights = {}
                
                with torch.no_grad():
                    target_logits = self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
                    task_accuracy = self.accuracy_fn(target_logits, target_labels).item()
                    accuracies_full.append(task_accuracy)
                    if ti < 50 and self.args.way <= 10:
                        self.metrics.plot_tsne(self.model.context_features, context_labels, self.model.classifier_params["class_prototypes"], "{}_tsne_initial_coreset_{:.3}".format(ti, task_accuracy))

                    del target_logits

                # Save the target/context features
                # We could optimize by only doing this if we're doing attention or divine selection, but for now whatever, it's a single forward pass
                with torch.no_grad():
                    self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
                    # Make a copy of the target features, otherwise we get a reference to what the model is storing
                    # (and we're about to send another task through it, so that's not what we want)
                    context_features, target_features = self.model.context_features.clone(), self.model.target_features.clone()
                
                for mode in ranking_modes:

                    if mode == 'loo':
                        weights_per_qp['loo'] = metaloo.calculate_loo(self.model, self.loss, context_images, context_labels, target_images, target_labels)
                    elif mode == 'attention':
                        # Make a copy of the attention weights otherwise we get a reference to what the model is storing
                        with torch.no_grad():
                            self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
                        attention_weights = self.model.attention_weights.copy()
                        weights_per_qp['attention'] = self.reshape_attention_weights(attention_weights, context_labels, len(target_labels)).cpu()
                        
                    elif mode == 'representer':
                        weights_per_qp['representer'] = metaloo.calculate_representer_values(self.model, self.loss, context_images, context_labels, context_features, target_labels, target_features, self.representer_args)
                    elif mode == 'random':
                        weights_per_qp['random'] = self.random_weights(context_images, context_labels, target_images, target_labels)
                    #rankings_per_qp[mode] = torch.argsort(weights_per_qp[mode], dim=1, descending=True)
                    # May want to normalize here:
                    weights[mode] = weights_per_qp[mode].sum(dim=0)
                    #rankings[mode] = torch.argsort(weights[mode], descending=True)
                    min_weight, max_weight = weights[mode].min(), weights[mode].max()
                    normalized_weights = (weights[mode] - min_weight)/(max_weight - min_weight)
                    for i in range(len(context_images)):
                        image_rankings = metaloo.add_image_rankings(image_rankings, task_dict["context_ids"][i], context_images[i], mode, normalized_weights[i])

            #import pdb; pdb.set_trace()
            self.metrics.plot_scatter(accuracies_full, item, 'Full Accuracy')

            # Righty-oh, now that we have our image_rankings, we need to do something with them.
            # Let's get some basic stats about them;
            # Aggregate to get indices.

            pickle.dump(image_rankings, open(os.path.join(self.args.checkpoint_dir, "rankings.pickle"), "wb"))
            
            for mode in ranking_modes:
                weights, image_ids = metaloo.weights_from_multirankings(image_rankings, mode)
                image_labels = self.dataset.get_labels_from_ids(image_ids)
                candidate_indices = self.select_top_k(self.args.top_k, weights, self.args.spread_constraint, image_labels)
                candidate_ids = image_ids[candidate_indices]
                candidate_labels = image_labels[candidate_indices]
                self.logger.log("{} candidate labels: {}".format(mode, candidate_labels))
                # Evaluate those indices:
                eval_accuracies = []
                for ti in tqdm(range(50), dynamic_ncols=True):
                    task_dict = self.dataset.get_task_from_ids(candidate_ids)
                    context_images, target_images, context_labels, target_labels = utils.prepare_task(task_dict, self.device)

                    with torch.no_grad():
                        target_logits = self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
                        task_accuracy = self.accuracy_fn(target_logits, target_labels).item()
                        eval_accuracies.append(task_accuracy)
                        if ti == 0:
                            self.metrics.plot_tsne(self.model.context_features, context_labels, self.model.classifier_params["class_prototypes"], "{}_tsne_candidate_coreset_{:.3}".format(mode, task_accuracy))
                            self.metrics.save_image_set(ti, context_images, "selected_{}".format(mode))

                        del target_logits
                    
                self.metrics.plot_scatter(eval_accuracies, item, 'Eval Accuracy ({})'.format(mode))

    def _funky_bimodal_inner_loop(self, context_images, context_labels, context_labels_orig, target_images, target_labels, ranking_modes, accuracies, num_unique_labels, ti, descrip):
        full_accuracy = -1
        rankings_per_qp = {}
        weights_per_qp = {}
        rankings = {}
        weights = {}
        
        with torch.no_grad():
            target_logits = self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
            full_accuracy = self.accuracy_fn(target_logits, target_labels).item()
            del target_logits
        
        # Save the target/context features
        # We could optimize by only doing this if we're doing attention or divine selection, but for now whatever, it's a single forward pass
        with torch.no_grad():
            self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
            # Make a copy of the target features, otherwise we get a reference to what the model is storing
            # (and we're about to send another task through it, so that's not what we want)
            context_features, target_features = self.model.context_features.clone(), self.model.target_features.clone()
        
        for mode in ranking_modes:

            if mode == 'loo':
                weights_per_qp['loo'] = metaloo.calculate_loo(self.model, self.loss, context_images, context_labels, target_images, target_labels)
            elif mode == 'attention':
                # Make a copy of the attention weights otherwise we get a reference to what the model is storing
                with torch.no_grad():
                    self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
                attention_weights = self.model.attention_weights.copy()
                weights_per_qp['attention'] = self.reshape_attention_weights(attention_weights, context_labels, len(target_labels)).cpu()
                
            elif mode == 'representer':
                weights_per_qp['representer'] = metaloo.calculate_representer_values(self.model, self.loss, context_images, context_labels, context_features, target_labels, target_features, self.representer_args)
            elif mode == 'random':
                weights_per_qp['random'] = self.random_weights(context_images, context_labels, target_images, target_labels)
            rankings_per_qp[mode] = torch.argsort(weights_per_qp[mode], dim=1, descending=True)
            # May want to normalize here:
            weights[mode] = weights_per_qp[mode].sum(dim=0)

        for key in weights.keys():
            if self.args.selection_mode == "top_k":
                candidate_indices = self.select_top_k(self.args.top_k, weights[key], context_labels, self.args.spread_constraint, )
            elif self.args.selection_mode == "multinomial":
                candidate_indices = metaloo.select_multinomial(self.args.top_k, weights[key], context_labels, self.args.spread_constraint)
            elif self.args.selection_mode == "divine":
                candidate_indices = metaloo.select_divine(self.args.top_k, weights[key], context_features, context_labels, self.args.spread_constraint, key)
            candidate_images, candidate_labels = context_images[candidate_indices], context_labels[candidate_indices]

            # If there aren't restrictions to ensure that every class is represented, we need to make special provision:
            reduced_candidate_labels, reduced_target_images, reduced_target_labels = metaloo.remove_unrepresented_points(candidate_labels, target_images, target_labels)

            num_unique_labels[key].append(len(context_labels_orig[candidate_indices].unique()))
            
            # Calculate accuracy on task using only selected candidates as context points
            with torch.no_grad():
                target_logits = self.model(candidate_images, reduced_candidate_labels, reduced_target_images, reduced_target_labels, MetaLearningState.META_TEST)
                task_accuracy = self.accuracy_fn(target_logits, reduced_target_labels).item()
                # Add the things that were incorrectly classified by default, because they weren't represented in the candidate context set
                task_accuracy = (task_accuracy * len(reduced_target_labels))/float(len(target_labels))
                accuracies[key].append(task_accuracy)
                    
                # Save out the selected candidates (?)

                if ti < 10:
                    self.metrics.save_image_set(ti, candidate_images, "candidate_{}_{}".format(descrip, key), labels=candidate_labels)
                    
        return full_accuracy, accuracies, num_unique_labels

    def funky_bimodal_task(self, path):
        self.logger.print_and_log("")  # add a blank line
        self.logger.print_and_log('Funky bimodal task using model {0:}: '.format(path))
        self.model = self.init_model()
        if path != 'None':
            self.model.load_state_dict(torch.load(path))
        for item in self.test_set:
            accuracies_full_bimodal = []
            accuracies_full_unimodal = []
            accuracies_bimodal = {} # per ranking-type
            accuracies_unimodal = {}
            if self.args.importance_mode == 'all':
                ranking_modes = ['loo', 'attention', 'representer', 'random']
            else:
                ranking_modes = [self.args.importance_mode]
                
            for mode in ranking_modes:
                accuracies_bimodal[mode] = []
                accuracies_unimodal[mode] = []
                
                
            # "accuracies_full_bimodal" will track the bimodal problem (i.e. reshaped with half as many classes)
            num_unique_labels_bimodal = {}
            num_unique_labels_unimodal = {}
            for mode in ranking_modes:
                num_unique_labels_bimodal[mode] = []
                num_unique_labels_unimodal[mode] = []

            for ti in tqdm(range(self.args.tasks), dynamic_ncols=True):
                task_dict = self.dataset.get_test_task(item)
                context_images, target_images, context_labels, target_labels = utils.prepare_task(task_dict, self.device)
                
                context_labels_orig, target_labels_orig = context_labels.clone(), target_labels.clone()
                context_labels = context_labels.floor_divide(2)
                target_labels = target_labels.floor_divide(2)
                
                if ti < 10:
                    self.metrics.save_image_set(ti, context_images, "context")
                    self.metrics.save_image_set(ti, target_images, "target")

                full_acc_bimodal, accuracies_bimodal, num_unique_labels_bimodal = self._funky_bimodal_inner_loop(context_images, context_labels, context_labels_orig, 
                                target_images, target_labels, ranking_modes, accuracies_bimodal, num_unique_labels_bimodal, ti, "bimodal")
                accuracies_full_bimodal.append(full_acc_bimodal)
                
                # Randomly drop one of the modes in target set
                orig_classes = target_labels_orig.unique()
                classes_to_keep = list(range(0, len(orig_classes), 2))
                keep_indices = []
                for c in classes_to_keep:
                    c_indices = utils.extract_class_indices(target_labels_orig, c)
                    keep_indices = keep_indices + c_indices.squeeze().tolist()
                keep_indices = np.array(keep_indices)
                new_target_images, new_target_labels = target_images[keep_indices], target_labels[keep_indices]

                if ti < 10:
                    self.metrics.save_image_set(ti, target_images[keep_indices], "unimodal_target", labels=target_labels_orig[keep_indices])
                
                full_acc_unimodal, accuracies_unimodal, num_unique_labels_unimodal = self._funky_bimodal_inner_loop(context_images, context_labels, context_labels_orig,
                                new_target_images, new_target_labels, ranking_modes, accuracies_unimodal, num_unique_labels_unimodal, ti, "unimodal")
                accuracies_full_unimodal.append(full_acc_unimodal)
                
            for key in num_unique_labels_bimodal.keys():
                unique_np = np.array(num_unique_labels_bimodal[key])
                self.logger.print_and_log("Average number of unique labels (bimodal) ({}): {}+/-{}".format(key, unique_np.mean(), unique_np.std()))
            self.metrics.plot_scatter(accuracies_full_bimodal, item, 'Accuracy (bimodal)')
            for key in accuracies_bimodal.keys():
                self.metrics.plot_scatter(accuracies_bimodal[key], item, 'Accuracy (bimodal) {}'.format(key))
                
            for key in num_unique_labels_unimodal.keys():
                unique_np = np.array(num_unique_labels_unimodal[key])
                self.logger.print_and_log("Average number of unique labels (unimodal) ({}): {}+/-{}".format(key, unique_np.mean(), unique_np.std()))
            self.metrics.plot_scatter(accuracies_full_unimodal, item, 'Accuracy (unimodal)')
            for key in accuracies_unimodal.keys():
                self.metrics.plot_scatter(accuracies_unimodal[key], item, 'Accuracy (unimodal) {}'.format(key))
           
    def select_shots(self, path):
        self.logger.print_and_log("")  # add a blank line
        self.logger.print_and_log('Selecting shots using model {0:}: '.format(path))
        self.model = self.init_model()
        if path != 'None':
            self.model.load_state_dict(torch.load(path))
        for item in self.test_set:
            accuracies_full = []
            accuracies = {}
            if self.args.importance_mode == 'all':
                ranking_modes = ['loo', 'attention', 'representer', 'random']
                correlations = {'attention_representer':[], 'attention_loo': [], 'representer_loo': [], 
                                    'attention_random': [], 'loo_random': [], 'representer_random': []}
                intersections = {'attention_representer':[], 'attention_loo': [], 'representer_loo': [], 
                                    'attention_random': [], 'loo_random': [], 'representer_random': []}
            else:
                ranking_modes = [self.args.importance_mode]
                
            for mode in ranking_modes:
                accuracies[mode] = []
                
            if self.args.test_case == "bimodal":
                # "accuracies_full" will track the bimodal problem (i.e. reshaped with half as many classes)
                # accuracies_orig will track accuracy w.r.t. the non bimodal problem
                accuracies_orig_full = []
                num_unique_labels = {}
                accuracies_orig_by_mode = {}
                for mode in ranking_modes:
                    num_unique_labels[mode] = []
                    accuracies_orig_by_mode[mode] = []

            for ti in tqdm(range(self.args.tasks), dynamic_ncols=True):
                task_dict = self.dataset.get_test_task(item)
                context_images, target_images, context_labels, target_labels = utils.prepare_task(task_dict, self.device)
                if self.args.test_case == "bimodal":
                    context_labels_orig, target_labels_orig = context_labels.clone(), target_labels.clone()
                    context_labels = context_labels.floor_divide(2)
                    target_labels = target_labels.floor_divide(2)

                rankings_per_qp = {}
                weights_per_qp = {}
                rankings = {}
                weights = {}
                
                if ti < 10 and self.args.way * self.args.shot <= 100:
                    self.metrics.save_image_set(ti, context_images, "context")
                    self.metrics.save_image_set(ti, target_images, "target")
                
                with torch.no_grad():
                    target_logits = self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
                    task_accuracy = self.accuracy_fn(target_logits, target_labels).item()
                    accuracies_full.append(task_accuracy)
                    del target_logits

                    if self.args.test_case == "bimodal":
                        target_logits = self.model(context_images, context_labels_orig, target_images, target_labels_orig, MetaLearningState.META_TEST)
                        task_accuracy = self.accuracy_fn(target_logits, target_labels_orig).item()
                        accuracies_orig_full.append(task_accuracy)
                
                # Save the target/context features
                # We could optimize by only doing this if we're doing attention or divine selection, but for now whatever, it's a single forward pass
                with torch.no_grad():
                    self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
                    # Make a copy of the target features, otherwise we get a reference to what the model is storing
                    # (and we're about to send another task through it, so that's not what we want)
                    context_features, target_features = self.model.context_features.clone(), self.model.target_features.clone()
                
                for mode in ranking_modes:

                    if mode == 'loo':
                        weights_per_qp['loo'] = metaloo.calculate_loo(self.model, self.loss, context_images, context_labels, target_images, target_labels)
                    elif mode == 'attention':
                        # Make a copy of the attention weights otherwise we get a reference to what the model is storing
                        with torch.no_grad():
                            self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
                        attention_weights = self.model.attention_weights.copy()
                        weights_per_qp['attention'] = self.reshape_attention_weights(attention_weights, context_labels, len(target_labels)).cpu()
                        
                    elif mode == 'representer':
                        weights_per_qp['representer'] = metaloo.calculate_representer_values(self.model, self.loss, context_images, context_labels, context_features, target_labels, target_features, self.representer_args)
                    elif mode == 'random':
                        weights_per_qp['random'] = self.random_weights(context_images, context_labels, target_images, target_labels)
                    rankings_per_qp[mode] = torch.argsort(weights_per_qp[mode], dim=1, descending=True)
                    # May want to normalize here:
                    weights[mode] = weights_per_qp[mode].sum(dim=0)
                    
                # Great, now we have the importance weights. Now what to do with them
                if self.args.importance_mode == 'all':
                    corr, inters = self.metrics.compare_rankings(rankings_per_qp['attention'], rankings_per_qp['representer'], "Attention vs Representer")
                    correlations['attention_representer'].append(corr)
                    intersections['attention_representer'].append(inters)
                    corr, inters = self.metrics.compare_rankings(rankings_per_qp['attention'], rankings_per_qp['loo'], "Attention vs Meta-LOO")
                    correlations['attention_loo'].append(corr)
                    intersections['attention_loo'].append(inters)
                    corr, inters = self.metrics.compare_rankings(rankings_per_qp['representer'], rankings_per_qp['loo'], "Representer vs Meta-LOO")
                    correlations['representer_loo'].append(corr) 
                    intersections['representer_loo'].append(inters)
                    corr, inters = self.metrics.compare_rankings(rankings_per_qp['attention'], rankings_per_qp['random'], "Attention vs Random")
                    correlations['attention_random'].append(corr)
                    intersections['attention_random'].append(inters)
                    corr, inters = self.metrics.compare_rankings(rankings_per_qp['loo'], rankings_per_qp['random'], "Meta-LOO vs Random")
                    correlations['loo_random'].append(corr)
                    intersections['loo_random'].append(inters)
                    corr, inters = self.metrics.compare_rankings(rankings_per_qp['representer'], rankings_per_qp['random'], "Representer vs Random")
                    correlations['representer_random'].append(corr) 
                    intersections['representer_random'].append(inters)

                for key in weights.keys():
                    if self.args.selection_mode == "top_k":
                        candidate_indices = self.select_top_k(self.args.top_k, weights[key], context_labels, self.args.spread_constraint)
                    elif self.args.selection_mode == "multinomial":
                        candidate_indices = metaloo.select_multinomial(self.args.top_k, weights[key], context_labels, self.args.spread_constraint)
                    elif self.args.selection_mode == "divine":
                        candidate_indices = metaloo.select_divine(self.args.top_k, weights[key], context_features, context_labels, self.args.spread_constraint, key)
                    candidate_images, candidate_labels = context_images[candidate_indices], context_labels[candidate_indices]

                    # If there aren't restrictions to ensure that every class is represented, we need to make special provision:
                    reduced_candidate_labels, reduced_target_images, reduced_target_labels = metaloo.remove_unrepresented_points(candidate_labels, target_images, target_labels)

                    if self.args.test_case == "bimodal":
                        num_unique_labels[key].append(len(context_labels_orig[candidate_indices].unique()))
                        candidate_labels_orig = context_labels_orig[candidate_indices]
                        reduced_candidate_labels_orig, reduced_target_images_orig, reduced_target_labels_orig = metaloo.remove_unrepresented_points(candidate_labels_orig, target_images, target_labels_orig)
                    
                    # Calculate accuracy on task using only selected candidates as context points
                    with torch.no_grad():
                        target_logits = self.model(candidate_images, reduced_candidate_labels, reduced_target_images, reduced_target_labels, MetaLearningState.META_TEST)
                        task_accuracy = self.accuracy_fn(target_logits, reduced_target_labels).item()
                        # Add the things that were incorrectly classified by default, because they weren't represented in the candidate context set
                        task_accuracy = (task_accuracy * len(reduced_target_labels))/float(len(target_labels))
                        accuracies[key].append(task_accuracy)
                            
                        # Save out the selected candidates (?)

                        if ti < 10 and self.args.way * self.args.shot <= 100:
                            self.metrics.save_image_set(ti, candidate_images, "candidate_{}_{}".format(ti, key), labels=candidate_labels)

                        if self.args.test_case == "bimodal":
                            target_logits = self.model(candidate_images, reduced_candidate_labels_orig, reduced_target_images_orig, reduced_target_labels_orig, MetaLearningState.META_TEST)
                            task_accuracy = self.accuracy_fn(target_logits, reduced_target_labels_orig).item()
                            task_accuracy = (task_accuracy * len(reduced_target_labels_orig))/float(len(target_labels_orig))
                            accuracies_orig_by_mode[key].append(task_accuracy)

                     
            if self.args.test_case == "bimodal":
                for key in num_unique_labels.keys():
                    unique_np = np.array(num_unique_labels[key])
                    self.logger.print_and_log("Average number of unique labels ({}): {}+/-{}".format(key, unique_np.mean(), unique_np.std()))

                self.metrics.plot_scatter(accuracies_orig_full, item, '*Accuracy (Not bimodal)')
                for key in accuracies_orig_by_mode.keys():
                    self.metrics.plot_scatter(accuracies_orig_by_mode[key], item, '*Accuracy (Not bimodal) {}'.format(key))
                
            self.metrics.plot_scatter(accuracies_full, item, 'Accuracy')

            for key in weights.keys():
                self.metrics.plot_scatter(accuracies[key], item, 'Accuracy {}'.format(key))
            if self.args.importance_mode == 'all':
                for key in correlations:
                    self.metrics.plot_scatter(correlations[key], item, 'Correlation {}'.format(key))
                for key in intersections:
                    self.metrics.plot_scatter(intersections[key], item, 'Intersections {}'.format(key))
        
    def make_noisy(self, task_dict):
        num_classes = len(np.unique(task_dict["target_labels"]))
        num_context_images = len(task_dict["context_labels"])
        
        if torch.is_tensor(task_dict["context_labels"]):
            new_context_labels = task_dict["context_labels"].clone()
        else:
            new_context_labels = task_dict["context_labels"].copy()
        if self.args.noise_type == "mislabel":
            num_noisy = max(1, int(self.args.error_rate * num_context_images))
            assert num_noisy < num_context_images
            # Select images from the context set randomly to be mislabeled 
            mislabeled_indices = rng.choice(num_context_images, num_noisy, replace=False)
            # Mislabels selected images randomly
            new_context_labels[mislabeled_indices] = (new_context_labels[mislabeled_indices] + rng.integers(1, num_classes, num_noisy)) % num_classes
        elif self.args.noise_type == "ood":
            # We want num_noisy/total patterns = error rate, where total patterns = num_clean + num_noisy
            # And num_clean = num_at_start/2
            assert num_classes % 2 == 0
            assert self.args.error_rate <= 0.5

            num_noisy = math.ceil(self.args.error_rate * ( num_context_images / 2.0) / (1 - self.args.error_rate))
            # Choose a class to drop
            classes_to_drop = rng.choice(num_classes, int(num_classes/2), replace=False)
            num_remaining_classes = num_classes - len(classes_to_drop)
            assert len(classes_to_drop) * self.args.shot >= num_noisy
            print("Resulting problem is approximately {}-way and {}-shot".format(num_remaining_classes, self.args.shot + num_noisy/num_remaining_classes))
            new_context_labels = task_dict["context_labels"].copy()
            wild_card_indices = np.empty(0, dtype=np.int8)
            for class_to_drop in classes_to_drop:
                # Build a list of all indices that we want to corrupt/drop
                wild_card_indices = np.append(wild_card_indices, utils.extract_class_indices(task_dict["context_labels"], class_to_drop))
            wild_card_indices = wild_card_indices.reshape(-1)
            np.random.shuffle(wild_card_indices)

            new_context_labels[wild_card_indices] = rng.integers(0, num_remaining_classes, len(wild_card_indices))
            #for index in wild_card_indices:
            #    new_context_labels[index] = rng.integers(0, num_remaining_classes, 1)[0]
            # Normalize labels
            clean_count = 0
            for c in range(num_classes):
                if c not in classes_to_drop:
                    c_indices = utils.extract_class_indices(task_dict["context_labels"], c)
                    new_context_labels[c_indices] = clean_count
                    clean_count += 1

            assert clean_count == num_remaining_classes
            assert num_noisy <= len(wild_card_indices)
            # Remove the rest from the context set
            if num_noisy < len(wild_card_indices):
                new_context_labels = np.delete(new_context_labels, wild_card_indices[num_noisy:])
                task_dict["context_images"] = np.delete(task_dict["context_images"], wild_card_indices[num_noisy:], axis=0)
                task_dict["context_labels"] = np.delete(task_dict["context_labels"], wild_card_indices[num_noisy:], axis=0)
                # Recalculate indices of mislabeled images by using the unnormalized context labels
                wild_card_indices = np.empty(0, dtype=np.int8)
                for class_to_drop in classes_to_drop:
                    wild_card_indices = np.append(wild_card_indices, utils.extract_class_indices(task_dict["context_labels"], class_to_drop))

            mislabeled_indices = wild_card_indices

            # Remove the dropped classes from the target set
            for class_to_drop in classes_to_drop:
                target_class_indices = utils.extract_class_indices(task_dict["target_labels"], class_to_drop)
                task_dict["target_images"] = np.delete(task_dict["target_images"], target_class_indices, axis=0)
                task_dict["target_labels"] = np.delete(task_dict["target_labels"], target_class_indices)

            # Normalize the target labels
            clean_count = 0
            for c in range(num_classes):
                if c not in classes_to_drop:
                    c_indices = utils.extract_class_indices(task_dict["target_labels"], c)
                    task_dict["target_labels"][c_indices] = clean_count
                    clean_count += 1

        task_dict["true_context_labels"] = task_dict["context_labels"]
        task_dict["context_labels"] = new_context_labels
        task_dict["noisy_context_indices"] = mislabeled_indices
        return task_dict
        
        
    def detect_noisy_shots(self, path):
        self.logger.print_and_log("")  # add a blank line
        self.logger.print_and_log('Detecting noisy shots using model {0:}: '.format(path))
        self.model = self.init_model()
        if path != 'None':
            self.model.load_state_dict(torch.load(path))


        for item in self.test_set:
            accuracies = {}
            overlaps = {}
            accuracies_clean = []
            accuracies_noisy = []
            
            if self.args.importance_mode == 'all':
                ranking_modes = ['loo', 'attention', 'representer', 'random']
            else:
                ranking_modes = [self.args.importance_mode]
                
            for mode in ranking_modes:
                accuracies[mode] = []
                overlaps[mode] = []
            for ti in tqdm(range(self.args.tasks), dynamic_ncols=True):
                task_dict = self.dataset.get_test_task(item)
                class_bins = list(range(0, self.args.way+1))

                # If mislabelling, then we calculate clean accuracy before doing any mislabeling
                if self.args.noise_type == "mislabel":
                    context_images, target_images, context_labels, target_labels = utils.prepare_task(task_dict, self.device)
                    # Calculate clean accuracy 
                    with torch.no_grad():
                        target_logits = self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
                        task_accuracy = self.accuracy_fn(target_logits, target_labels).item()
                        accuracies_clean.append(task_accuracy)

                        if ti<50:
                            initial_predictions = target_logits.argmax(axis=1)
                            self.metrics.plot_hist(initial_predictions, bins=class_bins, filename="class_distrib_initial", task_num=ti, title='Predicted classes (initial)', x_label='Predicted class label', density=True)
                            self.metrics.plot_tsne(self.model.context_features, context_labels, self.model.classifier_params["class_prototypes"], "{}_tsne_initial_{:.3}_acc".format(ti, task_accuracy))

                        del target_logits
                    del context_images
                    del target_images

                # Noisiness
                task_dict = self.make_noisy(task_dict)
                context_images, target_images, context_labels, target_labels = utils.prepare_task(task_dict, self.device)

                # For now, just don't bother with "clean" accuracy; it's hard to be comparable
                # If ood, then we need to first remove the superfluous class (i.e. make noisy), 
                # and then remove the noisy indices from the resulting context set
                #if self.args.noise_type == "ood":
                #    clean_context_images = context_images[task_dict["noisy_indices"]]
                #    clean_context_labels = context_labels

                with torch.no_grad():
                    target_logits = self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
                    task_accuracy = self.accuracy_fn(target_logits, target_labels).item()
                    accuracies_noisy.append(task_accuracy)

                    if ti<50:
                        noisy_predictions = target_logits.argmax(axis=1)
                        noisy_losses = self.loss(target_logits, target_labels, reduce=False)
                        self.metrics.plot_hist(noisy_predictions, bins=class_bins, filename="class_distrib_noisy", task_num=ti, title='Predicted classes (noisy)', x_label='Predicted class label', density=True)
                        self.metrics.plot_tsne(self.model.context_features, context_labels, self.model.classifier_params["class_prototypes"], "{}_tsne_noisy_{:.3}_acc".format(ti, task_accuracy))

                    del target_logits
                    gc.collect()

                rankings_per_qp = {}
                weights_per_qp = {}
                weights = {}

                if ti < 10 and self.args.way * self.args.shot <= 100:
                    self.metrics.save_image_set(ti, context_images, "context", labels=context_labels)
                    self.metrics.save_image_set(ti, target_images, "target", labels=target_labels)

                # Save the target/context features
                # We could optimize by only doing this if we're doing attention or divine selection, but for now whatever, it's a single forward pass
                # And we usually calculate attention weights
                with torch.no_grad():
                    self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
                    # Make a copy of the target features, otherwise we get a reference to what the model is storing
                    # (and we're about to send another task through it, so that's not what we want)
                    context_features, target_features = self.model.context_features.cpu(), self.model.target_features.cpu()                
                
                for mode in ranking_modes:
                    torch.cuda.empty_cache()
                    print(mode)
                    if mode == 'loo':
                        weights_per_qp['loo'] = metaloo.calculate_loo(self.model, self.loss, context_images, context_labels, target_images, target_labels)

                    elif mode == 'attention':
                        # Make a copy of the attention weights otherwise we get a reference to what the model is storing
                        with torch.no_grad():
                            self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
                        weights_per_qp['attention'] = self.reshape_attention_weights(self.model.attention_weights, context_labels, len(target_labels))
                        
                    elif mode == 'representer':
                        weights_per_qp['representer'] = metaloo.calculate_representer_values(self.model, self.loss, context_images, context_labels, context_features, target_labels, target_features, self.representer_args)
                    elif mode == 'random':
                        weights_per_qp['random'] = self.random_weights(context_images, context_labels, target_images, target_labels)
                    rankings_per_qp[mode] = torch.argsort(weights_per_qp[mode], dim=1, descending=True)
                    # May want to normalize here:
                    weights[mode] = weights_per_qp[mode].sum(dim=0)
                    
                for key in ranking_modes:
                    removed_indices_set = None
                    if self.args.selection_mode == "top_k":
                        candidate_indices = self.select_top_k(self.args.top_k, weights[key], context_labels, self.args.spread_constraint)
                    elif self.args.selection_mode == "multinomial":
                        candidate_indices = metaloo.select_multinomial(self.args.top_k, weights[key], context_labels, self.args.spread_constraint)
                    elif self.args.selection_mode == "divine":
                        candidate_indices = metaloo.select_divine(self.args.top_k, weights[key], context_features, context_labels, self.args.spread_constraint, key)
                    elif self.args.selection_mode == "drop":
                        candidate_indices, removed_indices = metaloo.select_by_dropping(weights[key], drop_rate=self.args.drop_rate)
                        removed_indices_set = set(removed_indices.tolist())
                    if removed_indices_set is None:
                        removed_indices_set = set(range(len(context_labels))).difference(set(candidate_indices.tolist()))

                    # Determine overlap between removed candidates and noisy shots
                    overlaps[key].append(float(len(removed_indices_set.intersection(set(task_dict["noisy_context_indices"]))))/float(len(task_dict["noisy_context_indices"])))
                    candidate_images, candidate_labels = context_images[candidate_indices], context_labels[candidate_indices]
                    
                    # If there aren't restrictions to ensure that every class is represented, we need to make special provision:
                    reduced_candidate_labels, reduced_target_images, reduced_target_labels = metaloo.remove_unrepresented_points(candidate_labels, target_images, target_labels)

                    # Calculate accuracy on task using only selected candidates as context points
                    with torch.no_grad():
                        target_logits = self.model(candidate_images, reduced_candidate_labels, reduced_target_images, reduced_target_labels, MetaLearningState.META_TEST)
                        task_accuracy = self.accuracy_fn(target_logits, reduced_target_labels).item()
                        # Add the things that were incorrectly classified by default, because they weren't represented in the candidate context set
                        task_accuracy = (task_accuracy * len(reduced_target_labels))/float(len(target_labels)) # TODO: We should add random accuracy back in, not zero.
                        accuracies[key].append(task_accuracy)

                        if ti<50:
                            red_predictions = target_logits.argmax(axis=1)
                            self.metrics.plot_hist(red_predictions, bins=class_bins, filename="class_distrib_reduced", task_num=ti, 
                                    title='Predicted classes (reduced, -{})'.format(self.args.way-len(reduced_candidate_labels.unique())), x_label='Predicted class label', density=True)
                            red_losses = self.loss(target_logits, target_labels, reduce=False)            
                            self.metrics.plot_tsne(self.model.context_features, candidate_labels, self.model.classifier_params["class_prototypes"], "{}_tsne_reduced_{:.3}_acc".format(ti, task_accuracy))

                            self.metrics.plot_scatter(noisy_losses, red_losses, x_label="Loss when context points flipped", y_label="Loss when selected context points are dropped", plot_title="Target Point Losses",
                                output_name="{}_target_scatter_dropped.png".format(ti), class_labels=reduced_target_labels, split_by_class_label=True)
                        del target_logits
                            
                        # Save out the selected candidates (?)
                    if ti < 50  and self.args.way * self.args.shot <= 100:
                        self.metrics.save_image_set(ti, context_images[removed_indices], "removed_by_{}".format(key), labels=context_labels[removed_indices])
                        self.metrics.plot_hist(task_dict["true_context_labels"][removed_indices], class_bins, "selected_label_hist_true", ti, 'Classes selected for dropping', 'True class label', 'Num points selected for drop')
                        self.metrics.plot_hist(context_labels[removed_indices], class_bins, "selected_label_hist_flipped", ti, 'Classes selected for dropping', 'Presented class label', 'Num points selected for drop')

            if len(accuracies_clean) > 0:
                self.metrics.plot_scatter(accuracies_clean, item, 'Clean Accuracy')
            self.metrics.plot_scatter(accuracies_noisy, item, 'Noisy Accuracy')

            for key in ranking_modes:
                self.metrics.plot_scatter(accuracies[key], item, 'Accuracy {}'.format(key))
                self.metrics.plot_scatter(overlaps[key], item, 'Shot Overlap {}'.format(key))


    def reshape_attention_weights(self, attention_weights, context_labels, num_target_points):
        weights_per_query_point = torch.zeros((num_target_points, len(context_labels)), device="cpu")
        # The method below of accessing attention weights per class is alright because it is how the model 
        # constructs this data structure.
        for ci, c in enumerate(torch.unique(context_labels)):
            c_indices = utils.extract_class_indices(context_labels, c)
            c_weights = attention_weights[ci].cpu().squeeze().clone()
            for q in range(c_weights.shape[0]):
                weights_per_query_point[q][c_indices] = c_weights[q]
                
        return weights_per_query_point


    def _augment(self, num_augment_per_class, images, labels):
        if num_augment_per_class < 1:
            return images, labels

        # horizontal flip
        transformed_images = tvf.hflip(images)
        accumulated_images = torch.cat((images, transformed_images))
        accumulated_labels = torch.cat((labels, labels))

        # resized crop
        if num_augment_per_class > 1:
            transformed_images = tvf.resized_crop(images, top=21, left=21, height=42, width=43, size=(84,84))
            accumulated_images = torch.cat((images, transformed_images))
            accumulated_labels = torch.cat((labels, labels))

        # grayscale
        if num_augment_per_class > 2:
            transformed_images = tvf.rgb_to_grayscale(images, num_output_channels=3)
            accumulated_images = torch.cat((images, transformed_images))
            accumulated_labels = torch.cat((labels, labels))

        # brighten
        if num_augment_per_class > 3:
            transformed_images = tvf.adjust_brightness(images, brightness_factor=1.2)
            accumulated_images = torch.cat((images, transformed_images))
            accumulated_labels = torch.cat((labels, labels))

        # darken
        if num_augment_per_class > 4:
            transformed_images = tvf.adjust_brightness(images, brightness_factor=0.8)
            accumulated_images = torch.cat((images, transformed_images))
            accumulated_labels = torch.cat((labels, labels))

        # up contrast
        if num_augment_per_class > 5:
            transformed_images = tvf.adjust_contrast(images, contrast_factor=1.2)
            accumulated_images = torch.cat((images, transformed_images))
            accumulated_labels = torch.cat((labels, labels))

        # down contrast
        if num_augment_per_class > 6:
            transformed_images = tvf.adjust_contrast(images, contrast_factor=0.8)
            accumulated_images = torch.cat((images, transformed_images))
            accumulated_labels = torch.cat((labels, labels))

        # shift hue 1
        if num_augment_per_class > 7:
            transformed_images = tvf.adjust_hue(images, hue_factor=0.1)
            accumulated_images = torch.cat((images, transformed_images))
            accumulated_labels = torch.cat((labels, labels))

        # shift hue 2
        if num_augment_per_class > 8:
            transformed_images = tvf.adjust_hue(images, hue_factor=-0.1)
            accumulated_images = torch.cat((images, transformed_images))
            accumulated_labels = torch.cat((labels, labels))

        # saturation
        if num_augment_per_class > 9:
            transformed_images = tvf.adjust_saturation(images, saturation_factor=1.2)
            accumulated_images = torch.cat((images, transformed_images))
            accumulated_labels = torch.cat((labels, labels))
        return accumulated_images, accumulated_labels

    def _set_images_to_black(self, context_images, context_labels, target_images, target_labels):
        num_context_images = context_images.size(0)
        num_target_images = target_images.size(0)
        for c in torch.unique(context_labels):
            is_first = True
            for i in range(num_context_images):
                if context_labels[i] == c:
                    if is_first:
                        is_first = False
                        for j in range(num_target_images):
                            if target_labels[j] == c:
                                context_images[i] = target_images[j]
                                break
                        continue
                    else:
                        context_images[i] = torch.randn(3, 84, 84)
                else:
                    continue

        return context_images

    def use_two_gpus(self):
        use_two_gpus = False
        if self.args.batch_normalization == "task_norm-i" and self.args.dataset == "meta-dataset" or\
                self.args.classifier == "protonets_cross_transformer" or\
                self.args.classifier == "versa_cross_transformer":
            use_two_gpus = True  # TaskNorm model does not fit on one GPU, so use model parallelism

        return use_two_gpus
        # return False

    def save_checkpoint(self, iteration):
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.validation_accuracies.get_current_best_accuracy_dict(),
        }, os.path.join(self.log_files.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.log_files.checkpoint_dir, 'checkpoint.pt'))
        self.start_iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.validation_accuracies.replace(checkpoint['best_accuracy'])

    def register_extra_parameters(self, model):
        if self.args.batch_normalization == "task_norm-i":
            for module in model.feature_extractor.modules():
                if isinstance(module, TaskNormI):
                    module.register_extra_weights()
            for module in model.set_encoder.modules():
                if isinstance(module, TaskNormI):
                    module.register_extra_weights()


if __name__ == "__main__":
    main()
