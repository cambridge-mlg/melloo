import torch
import torchvision.transforms.functional as tvf
import numpy as np
import argparse
import os
from tqdm import tqdm
from utils import Logger, LogFiles, ValidationAccuracies, cross_entropy_loss, categorical_accuracy, MetaLearningState, \
    coalesce_labels, merge_logits, mode_accuracy, extract_class_indices
from model import FewShotClassifier
from normalization_layers import TaskNormI
from dataset import get_dataset_reader
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

NUM_VALIDATION_TASKS = 200
NUM_TEST_TASKS = 600
PRINT_FREQUENCY = 1000


def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()
        self.log_files = LogFiles(self.args.checkpoint_dir, self.args.resume_from_checkpoint, self.args.mode == "test")
        self.logger = Logger(self.args.checkpoint_dir, "log.txt")

        self.logger.print_and_log("Options: %s\n" % self.args)
        self.logger.print_and_log("Checkpoint Directory: %s\n" % self.log_files.checkpoint_dir)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.train_set, self.validation_set, self.test_set = self.init_data()
        self.dataset = get_dataset_reader(
            args=self.args,
            train_set=self.train_set,
            validation_set=self.validation_set,
            test_set=self.test_set,
            device=self.device)

        self.loss = cross_entropy_loss
        self.accuracy_fn = categorical_accuracy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.validation_accuracies = ValidationAccuracies(self.validation_set)
        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()
        # Not sure whether to use shot or max_support_test
        self.top_k = min(self.args.top_k, self.args.shot)

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
                                                  "mnist", "cifar10", "cifar100"], default="meta-dataset",
                            help="Dataset to use.")
        parser.add_argument("--dataset_reader", choices=["official", "pytorch"], default="official",
                            help="Dataset reader to use.")
        parser.add_argument("--classifier", choices=["protonets_euclidean",
                                                     "protonets_attention",
                                                     "protonets_cross_transformer"],
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
        parser.add_argument("--max_support_train", type=int, default=500, help="Maximum support set size of meta-train task.")
        parser.add_argument("--max_support_test", type=int, default=500, help="Maximum support set size of meta-test task.")
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
        parser.add_argument("--l2_lambda", type=float, default=0.0001,
                            help="Value of lambda when doing L2 regularization on the classifier head")
        parser.add_argument("--l2_regularize_classifier", dest="l2_regularize_classifier", default=False,
                            action="store_true", help="If True, perform l2 regularization on the classifier head.")
        parser.add_argument("--top_k", type=int, default=2, help="How many points to retain from each class")
        parser.add_argument("--selection_mode", choices=["top_k", "multinomial"], default="top_k",
                            help="How to select the candidates from the importance weights")
        parser.add_argument("--importance_mode", choices=["attention", "loo", "representer", "all"], default="all",
                            help="How to calculate candidate importance")
        parser.add_argument("--kernel_agg", choices=["sum", "sum_abs", "class"], default="class",
                            help="When using representer importance, how to aggregate kernel information")
        

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
            self.generate_coreset(self.args.test_model_path)

    def get_from_gpu(self, value):
        if self.use_two_gpus():
            return value.cuda(0)
        else:
            return value

    def train_task(self, task_dict):
        context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)

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
                    context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)
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
            for _ in range(NUM_TEST_TASKS):
                task_dict = self.dataset.get_test_task(item)
                context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)
                # Do the forward pass with the target set so that we can get the feature embeddings
                with torch.no_grad():
                    target_logits = self.model(context_images, context_labels, target_images, target_labels,
                                               MetaLearningState.META_TEST)
                                               
                import pdb; pdb.set_trace()
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
                    c_indices = extract_class_indices(context_labels, c)
                    for i in c_indices:    
                        representer_tmp.append(representers[c][i])
                representer_per_query_point = torch.stack(representer_tmp).transpose(1,0)
                representer_approx = torch.matmul(alphas.transpose(1,0), torch.matmul(context_features, target_features.transpose(1,0)))
                representer_per_query_point = representer_per_query_point.cpu()
                
                representer_summed = representers.sum(dim=0).transpose(1,0).cpu()
                representer_abs_sum = representers.abs().sum(dim=0).transpose(1,0).cpu()
                import pdb; pdb.set_trace()
                
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

                ave_corr, ave_num_intersected = self.compare_rankings(weights_per_query_point, representer_per_query_point, "Attention vs Representer", weights=True)
                #ave_corr_mag, ave_num_intersected_mag = self.compare_rankings(weights_per_query_point, representer_per_query_point, "Attention vs Representer", weights=True)
                ave_corr_s, ave_num_intersected_s = self.compare_rankings(weights_per_query_point, representer_summed, "Attention vs Representer Summed", weights=True)
                #ave_corr_mag_s, ave_num_intersected_mag_s = self.compare_rankings(weights_per_query_point, representer_summed, "Attention vs Representer Summed", weights=True)
                ave_corr_abs_s, ave_num_intersected_abs_s = self.compare_rankings(weights_per_query_point, representer_abs_sum, "Attention vs Representer Sum Abs", weights=True)
                #ave_corr_mag_abs_s, ave_num_intersected_mag_abs_s = self.compare_rankings(weights_per_query_point, representer_abs_sum, "Attention vs Representer Sum Abs", weights=True)
                
            del target_logits

            accuracy = np.array(accuracies).mean() * 100.0
            accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

            self.logger.print_and_log('{0:}: {1:3.1f}+/-{2:2.1f}'.format(item, accuracy, accuracy_confidence))

    def calculate_representer_values(self, context_images, context_labels, context_features, target_labels, target_features):
        # Do the forward pass with the context set for the representer points:
        context_logits = self.model(context_images, context_labels, context_images, context_labels, MetaLearningState.META_TEST)
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
        
        kernels_agg = []
        for k in range(alphas.shape[1]):
            alphas_k = alphas_t[k]
            tmp_mat = alphas_k.unsqueeze(1).expand(len(context_labels), len(target_labels))
            kernels_k = tmp_mat * feature_prod
            kernels_agg.append(kernels_k.unsqueeze(0))
        representers = torch.cat(kernels_agg, dim=0)
        
        if self.args.kernel_agg == 'sum':
            representers_agg = representers.sum(dim=0).transpose(1,0).cpu()
        
        elif self.args.kernel_agg == 'sum_abs':
            representers_agg = representers.abs().sum(dim=0).transpose(1,0).cpu()
        
        elif self.args.kernel_agg == 'class':
            representer_tmp = []
            for c in torch.unique(context_labels):
                c_indices = extract_class_indices(context_labels, c)
                for i in c_indices:    
                    representer_tmp.append(representers[c][i])
            representers_agg = torch.stack(representer_tmp).transpose(1,0).cpu()
            
        else:
            print("Unsupported kernel aggregation method specified")
            return None
        return representers_agg

    def calculate_loo(self, context_images, context_labels, target_images, target_labels):
        return np.rand(len(target_labels), len(context_labels))

    def generate_coreset(self, path):
        self.logger.print_and_log("")  # add a blank line
        self.logger.print_and_log('Generating coreset using model {0:}: '.format(path))
        self.model = self.init_model()
        if path != 'None':
            self.model.load_state_dict(torch.load(path))

        for item in self.test_set:
            accuracies_full = []
            
            if self.args.importance_mode == 'all':
                ranking_modes = ['loo', 'attention', 'representer']
                correlations = {'attention_representer':[], 'attention_loo': [], 'representer_loo': []}
                intersections = {'attention_representer':[], 'attention_loo': [], 'representer_loo': []}
            else:
                ranking_modes = [self.args.importance_mode]
                
            for mode in ranking_modes:
                accuracies[mode] = []
            
            
            for _ in range(NUM_TEST_TASKS):
                task_dict = self.dataset.get_test_task(item)
                context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)
                import pdb; pdb.set_trace()
                rankings = {}
                
                #Both attention and representer require a forward pass with the context and target set to access the attention weights/feature embeddings
                # We also need to calculate the full task accuracy anyway
                with torch.no_grad():
                    target_logits = self.model(context_images, context_labels, target_images, target_labels, MetaLearningState.META_TEST)
                    task_accuracy = self.accuracy_fn(target_logits, target_labels)
                    accuracies_full.append(task_accuracy)
                    del target_logits
                
                for mode in ranking_modes:
                    if mode == 'loo':
                        rankings['loo'] = calculate_loo()
                    elif mode == 'attention' 
                        # Make a copy of the attention weights otherwise we get a reference to what the model is storing
                        attention_weights = self.model.attention_weights.clone()
                        rankings['attention'] = self.reshape_attention_weights(attention_weights, context_labels, len(target_labels)).cpu()
                        
                    elif mode == 'representer':
                        # Make a copy of the target features, otherwise we get a reference to what the model is storing
                        # (and we're about to send another task through it, so that's not what we want)
                        context_features, target_features = self.model.context_features.clone(), self.model.target_features.clone()
                        rankings['representer'] = self.calculate_representer_values(context_images, context_labels, context_features, target_labels, target_features)
                    rankings[mode] = torch.argsort(rankings[mode], dim=1, descending=True)
                    
                # Great, now we have the importance weights. Now what to do with them
                if self.args.importance_mode == 'all':
                    correlations['attention_representer'], intersections['attention_representer'] = compare_rankings(rankings['attention'], 
                                rankings['representer'], "Attention vs Representer")
                    correlations['attention_loo'], intersections['attention_representer'] = compare_rankings(rankings['attention'], 
                                rankings['loo'], "Attention vs Meta-LOO")
                    correlations['representer_loo'], intersections['attention_representer'] = compare_rankings(rankings['representer'], 
                                rankings['loo'], "Representer vs Meta-LOO")
                
                for key in rankings.keys():
                    if self.args.selection_mode == "top_k"
                        candidate_indices = self.select_top_k(rankings[key], context_labels)
                    elif self.args.selection_mode == "multinomial":
                        candidate_indices = self.select_multinomial(rankings[key], context_labels)
                    candidate_images, candidate_labels = context_images[candidate_indices], context_labels[candidate_indices]
                    
                    # Calculate accuracy on task using only selected candidates as context points
                    with torch.no_grad():
                        target_logits = self.model(candidate_images, candidate_labels, target_images, target_labels, MetaLearningState.META_TEST)
                        task_accuracy = self.accuracy_fn(target_logits, target_labels)
                        accuracies[key].append(task_accuracy)
                        
                    # Save out the selected candidates (?)

            self.print_and_log_metric(accuracies_full, item, 'Accuracy')
            for key in rankings.keys():
                self.print_and_log_metric(accuracies[key], item, 'Accuracy')
            if self.args.importance_mode == 'all':
                for key in correlations:
                    self.print_and_log_metric(correlations[key], item, 'Correlation')
                for key in intersections:
                    self.print_and_log_metric(intersections[key], item, 'Intersections')

    def select_top_k(self, ranking, class_labels):
        return ranking[0:self.args.topk]

    def select_multinomial(self, ranking, class_labels):
        import pdb; pdb.set_trace()
        
        
        return torch.multinomial(ranking, 2, replacement=False)


    def print_and_log_metric(self, values, item, metric_name="Accuracy"):
        metric = np.array(values).mean() * 100.0
        metric_confidence = (196.0 * np.array(values).std()) / np.sqrt(len(values))

        self.logger.print_and_log('{} {}: {:3.1f}+/-{:2.1f}'.format(metric_name, item, metric, metric_confidence))
    

    def compare_rankings(self, series1, series2, descriptor="", weights=False):
        if weights:
            ranking1 = torch.argsort(series1, dim=1, descending=True)
            ranking2 = torch.argsort(series2, dim=1, descending=True)
        else:
            ranking1, ranking2 = series1, series2
            
        ave_corr = 0.0
        ave_intersected = 0.0
        
        for t in range(ranking1.shape[0]):
            corr, p_value = kendalltau(ranking1[t], ranking2[t])
            ave_corr = ave_corr + corr
            
            top_k_1 = set(np.array(ranking1[t][1:self.args.topk]))
            top_k_2 = set(np.array(ranking2[t][1:self.args.topk]))
            intersected = sorted(top_k_1 & top_k_2)
            ave_intersected =  ave_intersected + len(intersected)
            
        ave_corr = ave_corr/ranking1.shape[0]
        ave_intersected = ave_intersected/ranking1.shape[0]
        
        #self.logger.print_and_log("Ave num intersected {}: {}".format(descriptor, ave_intersected))
        #self.logger.print_and_log("Ave corr {}: {}".format(descriptor, ave_corr))
        
        return ave_corr, ave_intersected

    def reshape_attention_weights(self, attention_weights, context_labels, num_target_points):
        weights_per_query_point = torch.zeros((num_target_points, len(context_labels)), device=self.device)
        for c in torch.unique(context_labels):
            c_indices = extract_class_indices(context_labels, c)
            c_weights = attention_weights[c].squeeze().clone()
            for q in range(c_weights.shape[0]):
                weights_per_query_point[q][c_indices] = c_weights[q]
                
        return weights_per_query_point

    def prepare_task(self, task_dict):
        if self.args.dataset_reader == "pytorch":
            return task_dict['context_images'], task_dict['target_images'],\
                   task_dict['context_labels'], task_dict['target_labels']
        else:
            context_images_np, context_labels_np = task_dict['context_images'], task_dict['context_labels']
            target_images_np, target_labels_np = task_dict['target_images'], task_dict['target_labels']

            context_images_np = context_images_np.transpose([0, 3, 1, 2])
            context_images_np, context_labels_np = self.shuffle(context_images_np, context_labels_np)
            context_images = torch.from_numpy(context_images_np)
            context_labels = torch.from_numpy(context_labels_np)

            target_images_np = target_images_np.transpose([0, 3, 1, 2])
            target_images_np, target_labels_np = self.shuffle(target_images_np, target_labels_np)
            target_images = torch.from_numpy(target_images_np)
            target_labels = torch.from_numpy(target_labels_np)

            context_images = context_images.to(self.device)
            target_images = target_images.to(self.device)
            context_labels = context_labels.type(torch.LongTensor).to(self.device)
            target_labels = target_labels.type(torch.LongTensor).to(self.device)

            return context_images, target_images, context_labels, target_labels

    def slim_images_and_labels(self, images, labels):
        slimmed_images = []
        slimmed_labels = []
        for c in torch.unique(labels):
            slimmed_labels.append(c)
            class_images = torch.index_select(images, 0, extract_class_indices(labels, c))
            slimmed_images.append(class_images[0])

        return torch.stack(slimmed_images), torch.stack(slimmed_labels)

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

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]

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
