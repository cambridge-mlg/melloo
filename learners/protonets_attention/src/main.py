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
            self.test(self.args.test_model_path)

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
                target_logits = self.model(context_images, context_labels, target_images, target_labels,
                                           MetaLearningState.META_TEST)
                
                task_loss = self.loss(target_logits, target_labels)
                regularization_term = (self.model.feature_adaptation_network.regularization_term())
                regularizer_scaling = 0.001
                task_loss += regularizer_scaling * regularization_term

                if self.args.l2_regularize_classifier:
                    classifier_regularization_term = self.model.classifer_regularization_term()
                    task_loss += self.args.l2_lambda * classifier_regularization_term
                torch.autograd.grad(task_loss, self.model.context_features, retain_graph=True)
                                           
                                           
                accuracy = self.accuracy_fn(target_logits, target_labels)
                accuracies.append(accuracy.item())

                context_features, target_features, attention_weights = self.model.context_features, self.model.target_features, self.model.attention_weights
                
                weights_per_context_point = torch.zeros((len(target_labels), len(context_labels)), device=self.device)
                for c in torch.unique(context_labels):
                    c_indices = extract_class_indices(context_labels, c)
                    c_weights = attention_weights[c].squeeze()
                    for q in range(c_weights.shape[0]):
                        weights_per_context_point[q][c_indices] = c_weights[q]
                import pdb; pdb.set_trace()

                rankings = torch.argsort(weights_per_context_point, dim=1, descending=True)

            del target_logits

            accuracy = np.array(accuracies).mean() * 100.0
            accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

            self.logger.print_and_log('{0:}: {1:3.1f}+/-{2:2.1f}'.format(item, accuracy, accuracy_confidence))

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
            context_labels = context_labels.to(self.device)
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
