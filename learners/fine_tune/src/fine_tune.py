import torch
import numpy as np
import argparse
import os
from tqdm import tqdm
from utils import Logger, accuracy
from model import FineTuner
from meta_dataset_reader import MetaDatasetReader, SingleDatasetReader
from attacks.attack_utils import AdversarialDataset
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings
import tensorflow as tf
from attacks.attack_utils import split_target_set, make_adversarial_task_dict, make_swap_attack_task_dict, infer_num_shots
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Quiet TensorFlow warningsimport globals
from attacks.attack_helpers import create_attack

def main():
    learner = Learner()
    learner.run()


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

class Learner:
    def __init__(self):
        self.args = self.parse_command_line()
        self.logger = Logger(self.args.checkpoint_dir, self.args.log_file)

        self.logger.print_and_log("Options: %s\n" % self.args)
        self.logger.print_and_log("Checkpoint Directory: %s\n" % self.args.checkpoint_dir)

        gpu_device = 'cuda:0'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.train_set, self.validation_set, self.test_set = self.init_data()
        
        assert self.args.target_set_size_multiplier >= 1
        num_target_sets = self.args.target_set_size_multiplier
        if self.args.indep_eval or self.args.attack_mode == 'swap':
            num_target_sets += self.args.num_indep_eval_sets
        if self.args.dataset == "meta-dataset":
            if self.args.query_test * self.args.target_set_size_multiplier > 50:
                self.logger.print_and_log("WARNING: Very high number of query points requested. Query points = query_test * target_set_size_multiplier = {} * {} = {}".format(self.args.query_test, self.args.target_set_size_multiplier, self.args.query_test * self.args.target_set_size_multiplier))

            self.dataset = MetaDatasetReader(self.args.data_path, "attack", self.train_set, self.validation_set,
                                             self.test_set, self.args.max_way_train, self.args.max_way_test,
                                             self.args.max_support_train, self.args.max_support_test, self.args.query_test * num_target_sets)
        elif self.args.dataset != "from_file":
            self.dataset = SingleDatasetReader(self.args.data_path, "attack", self.args.dataset, self.args.way,
                                               self.args.shot, self.args.query_train, self.args.query_test * num_target_sets)
        else:
            self.dataset = AdversarialDataset(self.args.data_path)
            self.max_test_tasks = min(self.dataset.get_num_tasks(), self.args.test_tasks)
        self.accuracy_fn = accuracy

    def init_model(self):
        model = FineTuner(args=self.args, device=self.device)
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
        parser.add_argument("--data_path", default="../datasets", help="Path to dataset records.")
        parser.add_argument("--dataset", choices=["meta-dataset", "ilsvrc_2012", "omniglot", "aircraft", "cu_birds",
                                                  "dtd", "quickdraw", "fungi", "vgg_flower", "traffic_sign", "mscoco",
                                                  "mnist", "cifar10", "cifar100", "from_file"], default="meta-dataset",
                            help="Dataset to use.")

        parser.add_argument('--test_datasets', nargs='+', help='Datasets to use for testing',
                            default=["quickdraw", "ilsvrc_2012", "omniglot", "aircraft", "cu_birds", "dtd",     "fungi",
                                     "vgg_flower", "traffic_sign", "mscoco", "mnist", "cifar10", "cifar100"])
        parser.add_argument("--feature_extractor", choices=["mnasnet", "resnet", "vgg11", "resnet18", "resnet34",
                                                            "maml_convnet", "protonets_convnet"], default="mnasnet",
                            help="Dataset to use.")
        parser.add_argument("--pretrained_feature_extractor_path", default="./learners/fine_tune/models/pretrained_mnasnet.pth",
                            help="Path to pretrained feature extractor model.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.05, help="Learning rate.")
        parser.add_argument("--weight_decay", "-wd", type=float, default=0.001, help="Weight decay.")
        parser.add_argument("--regularizer_scaling", type=float, default=0.001, help="Scaling for FiLM layer regularization.")
        parser.add_argument("--checkpoint_dir", "-c", default='./checkpoints', help="Directory to save checkpoint to.")
        parser.add_argument("--feature_adaptation", choices=["no_adaptation", "film"], default="film",
                            help="Method to adapt feature extractor parameters.")
        parser.add_argument("--iterations", "-i", type=int, default=50, help="Number of fine-tune iterations.")
        
        parser.add_argument("--max_way_train", type=int, default=40,
                            help="Maximum way of meta-dataset meta-train task.")
        parser.add_argument("--max_way_test", type=int, default=50, help="Maximum way of meta-dataset meta-test task.")
        parser.add_argument("--max_support_train", type=int, default=400,
                            help="Maximum support set size of meta-dataset meta-train task.")
        parser.add_argument("--max_support_test", type=int, default=400,
                            help="Maximum support set size of meta-dataset meta-test task.")
        parser.add_argument("--way", type=int, default=5, help="Way of single dataset task.")
        parser.add_argument("--shot", type=int, default=1, help="Shots per class for context of single dataset task.")
        parser.add_argument("--query_train", type=int, default=10,
                            help="Shots per class for target  of single dataset task.")
        parser.add_argument("--query_test", type=int, default=10,
                            help="Shots per class for target  of single dataset task.")                            
        parser.add_argument("--test_tasks", "-t", type=int, default=1000, help="Number of tasks to test for each dataset.")
        parser.add_argument("--batch_size", "-b", type=int, default=1000, help="Batch size.")
        parser.add_argument("--log_file", default="log.tx", help="Name of log file")
        parser.add_argument("--attack_mode", choices=["context", "target", "swap"], default="context",
                            help="Type of attack being transferred")
        parser.add_argument("--target_set_size_multiplier", type=int, default=1,
                            help="For swap attacks, the relative size of the target set used when generating the adv context set (eg. x times larger). Currently only implemented for swap attacks") 
        parser.add_argument("--indep_eval", default=False,
                            help="Whether to use independent target sets for evaluation automagically")     
        parser.add_argument("--num_indep_eval_sets", type=int, default=50,
                            help="Number of independent datasets to use for evaluation")          
        parser.add_argument("--attack_tasks", "-a", type=int, default=10,
                            help="Number of tasks when performing attack.")
        parser.add_argument("--attack_config_path", help="Path to attack config file in yaml format.")    
        parser.add_argument("--continue_from_task", type=int, default=0,
                            help="When saving out large numbers of tasks one by one, this allows us to continue labelling new tasks from a certain point")
        args = parser.parse_args()

        return args

    def run(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session:
            if self.args.dataset == 'from_file':
                self.finetune(session)
            else:
                self.attack(session)

    def eval(self, task_index):
        eval_acc = []
        target_image_sets, target_label_sets = self.dataset.get_eval_task(task_index, self.device)
        for s in range(len(target_image_sets)):
            accuracy = self.model.test_linear(target_image_sets[s], target_label_sets[s])
            eval_acc.append(accuracy)
        return np.array(eval_acc).mean()

    def calc_accuracy(self, context_images, context_labels, target_images, target_labels):
        self.model.fine_tune(context_images, context_labels)
        accuracy = self.model.test_linear(target_images, target_labels)
        return accuracy

    def print_average_accuracy(self, accuracies, descriptor):
        accuracy = np.array(accuracies).mean() * 100.0
        accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
        self.logger.print_and_log('{0:} {1:3.1f}+/-{2:2.1f}'.format(descriptor, accuracy, accuracy_confidence))

    def attack(self, session):
        self.logger.print_and_log("Finetuning on data found in {} ({})".format(self.args.data_path, self.args.dataset))
        self.logger.print_and_log("using feature extractor from {}".format(self.args.pretrained_feature_extractor_path))
        # Swap attacks only make sense if doing evaluation with independent target sets
        #
        #self.model = self.init_model()
        #self.model.load_state_dict(torch.load(path), strict=False)

        context_attack = create_attack(self.args.attack_config_path, self.args.checkpoint_dir)
        context_attack.set_attack_mode('context')
        assert self.args.indep_eval

        target_attack = create_attack(self.args.attack_config_path, self.args.checkpoint_dir)
        target_attack.set_attack_mode('target')

        for item in self.test_set:
            # Accuracies for setting in which we generate attacks.
            # Useful for debugging attacks
            gen_clean_accuracies = []
            gen_adv_context_accuracies = []
            gen_adv_target_accuracies = []

            # Accuracies for evaluation setting
            clean_accuracies = []
            clean_target_as_context_accuracies = []
            adv_context_accuracies = []
            adv_target_as_context_accuracies = []

            for t in tqdm(range(self.args.attack_tasks - self.args.continue_from_task), dynamic_ncols=True):
                task_dict = self.dataset.get_test_task(item, session)
                if self.args.continue_from_task != 0:
                    #Skip the first one, which is deterministic
                    task_dict = self.dataset.get_test_task(item, session)
                context_images, target_images, context_labels, target_labels, extra_datasets = self.prepare_task(task_dict)
                target_images_small, target_labels_small, eval_images, eval_labels = extra_datasets

                adv_context_images, adv_context_indices = context_attack.generate(context_images, context_labels, target_images, target_labels, self.model, self.model.forward, self.model.device)
                adv_target_images, adv_target_indices = target_attack.generate(context_images, context_labels, target_images_small, target_labels_small, self.model, self.model.forward, self.model.device)

                adv_target_as_context = context_images.clone()
                # Parallel array to keep track of where we actually put the adv_target_images
                # Since not all of them might have room to get swapped
                swap_indices_context = []
                swap_indices_adv = []
                target_labels_int = target_labels.type(torch.IntTensor)
                failed_to_swap = 0

                for index in adv_target_indices:
                    c = target_labels_int[index]
                    # Replace the first best instance of class c with the adv query point (assuming we haven't already swapped it)
                    shot_indices = extract_class_indices(context_labels.cpu(), c)
                    k = 0
                    while k < len(shot_indices) and shot_indices[k] in swap_indices_context:
                        k += 1
                    if k == len(shot_indices):
                        failed_to_swap += 1
                    else:
                        index_to_swap = shot_indices[k]
                        swap_indices_context.append(index_to_swap)
                        swap_indices_adv.append(index)
                assert (len(swap_indices_context)+failed_to_swap) == len(adv_target_indices)

                # First swap in the clean targets, to make sure the two clean accs are the same (debug)
                for i, swap_i in enumerate(swap_indices_context):
                    adv_target_as_context[swap_i] = target_images[swap_indices_adv[i]]

                with torch.no_grad():
                    # Evaluate in normal/generation setting
                    gen_clean_accuracies.append(
                        self.calc_accuracy(context_images, context_labels, target_images, target_labels))
                    gen_adv_context_accuracies.append(
                        self.calc_accuracy(adv_context_images, context_labels, target_images, target_labels))
                    gen_adv_target_accuracies.append(
                        self.calc_accuracy(context_images, context_labels, adv_target_images, target_labels_small))

                    # Evaluate on independent target sets
                    for k in range(len(eval_images)):
                        eval_imgs_k = eval_images[k].to(self.device)
                        eval_labels_k = eval_labels[k].to(self.device)
                        clean_accuracies.append(self.calc_accuracy(context_images, context_labels, eval_imgs_k, eval_labels_k))
                        clean_target_as_context_accuracies.append(self.calc_accuracy(adv_target_as_context, context_labels, eval_imgs_k, eval_labels_k))
                    
                    for k in range(len(eval_images)):
                        eval_imgs_k = eval_images[k].to(self.device)
                        eval_labels_k = eval_labels[k].to(self.device)

                        adv_context_accuracies.append(self.calc_accuracy(adv_context_images, context_labels, eval_imgs_k, eval_labels_k))
                        # Now swap in the adv targets
                        for i, swap_i in enumerate(swap_indices_context):
                            adv_target_as_context[swap_i] = adv_target_images[swap_indices_adv[i]]
                        adv_target_as_context_accuracies.append(self.calc_accuracy(adv_target_as_context, context_labels, eval_imgs_k, eval_labels_k))

                del adv_target_as_context
                '''
                if self.args.save_attack:
                    adv_task_dict = make_swap_attack_task_dict(context_images, context_labels, target_images_small,
                                                               target_labels_small,
                                                               adv_context_images, adv_context_indices,
                                                               adv_target_images, adv_target_indices,
                                                               self.args.way, self.args.shot, self.args.query_test,
                                                               eval_images, eval_labels)
                    #if self.args.continue_from_task != 0:
                    save_partial_pickle(os.path.join(self.args.checkpoint_dir, "adv_task"), t+self.args.continue_from_task, adv_task_dict)
                '''
                del adv_context_images, adv_target_images
                
            self.print_average_accuracy(gen_clean_accuracies, "Gen setting: Clean accuracy")
            self.print_average_accuracy(gen_adv_context_accuracies, "Gen setting: Context attack accuracy")
            self.print_average_accuracy(gen_adv_target_accuracies, "Gen setting: Target attack accuracy")

            self.print_average_accuracy(clean_accuracies, "Clean accuracy")
            self.print_average_accuracy(clean_target_as_context_accuracies, "Clean Target as Context accuracy")
            self.print_average_accuracy(adv_context_accuracies, "Context attack accuracy")
            self.print_average_accuracy(adv_target_as_context_accuracies, "Adv Target as Context accuracy")
            #self.logger.print_and_log('Average number of eval tests over all tasks {0:3.1f}'.format(ave_num_eval_sets/float(self.args.attack_tasks)))


    def finetune(self, session):
        self.logger.print_and_log("")  # add a blank line
        self.logger.print_and_log("Finetuning on data found in {}".format(self.args.data_path))
        self.logger.print_and_log("using feature extractor from {}".format(self.args.pretrained_feature_extractor_path))

        with torch.no_grad():
            accuracies = {'clean_0': [], 'clean': [], 'adv_target_0': [], 'adv_context_0': [],
                             'adv_context': [], 'target_swap_0': [], 'target_swap_1': [], 'target_swap': []}
            attack_descrips = {'clean_0': 'Clean Acc (gen setting)', 'clean': 'Clean Acc', 'adv_target_0': 'Target Attack Acc (gen setting)',
                               'adv_context_0': 'Context Attack Acc (gen setting)', 'adv_context': 'Context Attack Acc',
                               'target_swap_0': 'Target as Context (gen setting)', 'target_swap_1': 'Target as Context (straight)', 'target_swap': 'Target as Context'}
            ordered_keys = ['clean_0', 'adv_target_0', 'adv_context_0', 'target_swap_0', 'target_swap_1', 'clean', 'adv_context', 'target_swap']

            for task in tqdm(range(self.max_test_tasks),dynamic_ncols=True):
                # Clean task
                context_images, context_labels, target_images, target_labels = self.dataset.get_clean_task(task, self.device)
                # fine tune the model to the current task
                self.model.fine_tune(context_images, context_labels)
                accuracy = self.model.test_linear(target_images, target_labels)
                accuracies['clean_0'].append(accuracy)
                accuracies['clean'].append(self.eval(task))

                if self.args.attack_mode == "target":
                    # Run test for efficacy of adversarial target points
                    # Since we don't need to retrain for a target attack, we can re-use the model we just learned
                    context_images, context_labels, adv_target_images, target_labels = self.dataset.get_adversarial_task(task, self.device)
                    accuracy = self.model.test_linear(adv_target_images, target_labels)
                    accuracies['adv_target_0'].append(accuracy)

                elif self.args.attack_mode == "context":
                    context_images, context_labels, target_images, target_labels = self.dataset.get_adversarial_task(task, self.device)
                    # fine tune the model to the current task
                    self.model.fine_tune(context_images, context_labels)
                    accuracy = self.model.test_linear(target_images, target_labels)
                    accuracies['adv_context_0'].append(accuracy)
                    accuracies['adv_context'].append(self.eval(task))

                else: #Swap attack
                    # First run target test
                    _, _, adv_target_images, target_labels = self.dataset.get_adversarial_task(task, self.device, swap_mode="target")
                    accuracy = self.model.test_linear(adv_target_images, target_labels)
                    accuracies['adv_target_0'].append(accuracy)

                    # Now use the adv target set as a context set
                    self.model.fine_tune(adv_target_images, target_labels)
                    accuracy = self.model.test_linear(context_images, context_labels)
                    accuracies['target_swap_0'].append(accuracy)
                    accuracies['target_swap_1'].append(self.eval(task)) # will this work?

                    # Then request the adversarial context set as usual, to run a context attack
                    context_images, context_labels, target_images, target_labels = self.dataset.get_adversarial_task(task, self.device, swap_mode="context")

                    # fine tune the model to the current task
                    self.model.fine_tune(context_images, context_labels)
                    accuracy = self.model.test_linear(target_images, target_labels)
                    accuracies['adv_context_0'].append(accuracy)
                    accuracies['adv_context'].append(self.eval(task))

                    target_as_context, context_labels = self.dataset.get_adversarial_task(task, self.device, swap_mode="target_as_context")
                    self.model.fine_tune(target_as_context, context_labels)
                    accuracies['target_swap'].append(self.eval(task)) # will this work?

            for key in ordered_keys:
                if len(accuracies[key]) > 0:
                    self.print_average_accuracy(accuracies[key], attack_descrips[key])

    def prepare_task(self, task_dict):
        context_images_np, context_labels_np = task_dict['context_images'], task_dict['context_labels']
        target_images_np, target_labels_np = task_dict['target_images'], task_dict['target_labels']

        context_images_np = context_images_np.transpose([0, 3, 1, 2])
        context_images_np, context_labels_np = self.shuffle(context_images_np, context_labels_np)
        context_images = torch.from_numpy(context_images_np)
        context_labels = torch.from_numpy(context_labels_np)
        context_labels = context_labels.type(torch.LongTensor)

        all_target_images_np = target_images_np.transpose([0, 3, 1, 2])
        all_target_images_np, target_labels_np = self.shuffle(all_target_images_np, target_labels_np)
        all_target_images = torch.from_numpy(all_target_images_np)
        all_target_labels = torch.from_numpy(target_labels_np)
        all_target_labels = all_target_labels.type(torch.LongTensor)

        # Target set size == context set size, no extra pattern requested for eval, no worries.
        if self.args.target_set_size_multiplier == 1 and not self.args.indep_eval:
            target_images, target_labels = all_target_images, all_target_labels
            target_images_np = all_target_images_np
            extra_datasets = (context_images_np, target_images_np, None, None)
        else:
            # Split the larger set of target images/labels up into smaller sets of appropriate shot and way
            # This is slightly trickier for meta-dataset
            if self.args.dataset == "meta-dataset":
                target_set_shot = self.args.query_test
                task_way = len(torch.unique(context_labels))
                if self.args.target_set_size_multiplier * target_set_shot * task_way > all_target_images.shape[0]:
                    # Check the actual target set's shots can be inferred/is what we expect
                    target_set_shot = infer_num_shots(all_target_labels)
                    assert target_set_shot != -1
                    num_target_sets = all_target_images.shape[0] / (task_way * target_set_shot)
                    self.logger.print_and_log("Task had insufficient data for requested number of eval sets. Using what's available: {}".format(num_target_sets))
            else:
                target_set_shot = self.args.shot
                task_way = self.args.way
                assert self.args.target_set_size_multiplier * target_set_shot * task_way <= all_target_images.shape[0]

            # If this is a swap attack, then we need slightly different results from the target set splitter
            if self.args.attack_mode != 'swap':
                target_images, target_labels, eval_images, eval_labels, target_images_np = split_target_set(
                    all_target_images, all_target_labels, self.args.target_set_size_multiplier, target_set_shot,
                    all_target_images_np=all_target_images_np)
                extra_datasets = (context_images_np, target_images_np, eval_images, eval_labels)
            else:
                target_images, target_labels, eval_images, eval_labels, target_images_small, target_labels_small = split_target_set(
                    all_target_images, all_target_labels, self.args.target_set_size_multiplier, target_set_shot,
                    return_first_target_set=True)
                target_images_small = target_images_small.to(self.device)
                target_labels_small = target_labels_small.to(self.device)
                extra_datasets = (target_images_small, target_labels_small, eval_images, eval_labels)

        context_images = context_images.to(self.device)
        target_images = target_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.to(self.device)

        return context_images, target_images, context_labels, target_labels, extra_datasets

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]


if __name__ == "__main__":
    main()
