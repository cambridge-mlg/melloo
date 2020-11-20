import torch
import numpy as np
import argparse
import os
from learners.protonets.src.utils import print_and_log, get_log_files, categorical_accuracy, loss
from learners.protonets.src.model import ProtoNets
from learners.protonets.src.data import MiniImageNetData, OmniglotData

NUM_VALIDATION_TASKS = 400
NUM_TEST_TASKS = 1
PRINT_FREQUENCY = 100


def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.resume_from_checkpoint, self.args.mode == "test")

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        gpu_device = 'cuda:0'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.loss = loss

        if self.args.dataset == "mini_imagenet":
            self.dataset = MiniImageNetData(self.args.data_path, 111)
        elif self.args.dataset == "omniglot":
            self.dataset = OmniglotData(self.args.data_path, 111)
        else:
            self.dataset = None

        self.accuracy_fn = categorical_accuracy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()
        self.best_validation_accuracy = 0.0

    def init_model(self):
        model = ProtoNets(args=self.args).to(self.device)
        model.train()
        return model

    """
    Command line parser
    """
    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset", choices=['omniglot', "mini_imagenet"], default="mini_imagenet",
                            help="Dataset to use.")
        parser.add_argument("--data_path", default="../datasets", help="Path to dataset records.")
        parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test",
                            help="Whether to run training only, testing only, or both training and testing.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=1,
                            help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default='../checkpoints', help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--training_iterations", "-i", type=int, default=20000,
                            help="Number of meta-training iterations.")
        parser.add_argument("--val_freq", type=int, default=200, help="Number of iterations between validations.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False,
                            action="store_true", help="Restart from latest checkpoint.")
        parser.add_argument("--train_way", type=int, default=20, help="Way of meta-train task.")
        parser.add_argument("--train_shot", type=int, default=5, help="Shots per class for meta-train context sets.")
        parser.add_argument("--test_way", type=int, default=5, help="Way of meta-test task.")
        parser.add_argument("--test_shot", type=int, default=5, help="Shots per class for meta-test context sets.")
        parser.add_argument("--query", type=int, default=15, help="Shots per class for target")

        args = parser.parse_args()

        return args

    def run(self):
        if self.args.mode == 'train' or self.args.mode == 'train_test':
            train_accuracies = []
            losses = []
            total_iterations = self.args.training_iterations
            for iteration in range(self.start_iteration, total_iterations):
                current_lr = self.adjust_learning_rate(iteration)
                # torch.set_grad_enabled(True)
                task_dict = self.dataset.get_train_task(self.args.train_way,
                                                        self.args.train_shot,
                                                        self.args.query)
                task_loss, task_accuracy = self.train_task(task_dict)
                losses.append(task_loss)
                train_accuracies.append(task_accuracy)

                # optimize
                if ((iteration + 1) % self.args.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if (iteration + 1) % PRINT_FREQUENCY == 0:
                    # print training stats
                    print_and_log(self.logfile,'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}, lr: {:.7f}'
                                  .format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
                                          torch.Tensor(train_accuracies).mean().item(), current_lr))
                    train_accuracies = []
                    losses = []

                if ((iteration + 1) % self.args.val_freq == 0) and (iteration + 1) != total_iterations:
                    # validate
                    accuracy = self.validate()
                    # save the model if validation is the best so far
                    if accuracy > self.best_validation_accuracy:
                        self.best_validation_accuracy = accuracy
                        torch.save(self.model.state_dict(), self.checkpoint_path_validation)
                        print_and_log(self.logfile, 'Best validation model was updated.')
                        print_and_log(self.logfile, '')
                    self.save_checkpoint(iteration + 1)

            # save the final model
            torch.save(self.model.state_dict(), self.checkpoint_path_final)

        if self.args.mode == 'train_test':
            self.test(self.checkpoint_path_final)
            self.test(self.checkpoint_path_validation)

        if self.args.mode == 'test':
            self.test_leave_one_out(self.args.test_model_path)

        self.logfile.close()

    def train_task(self, task_dict):
        self.model.train()
        context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict, shuffle=False)

        logits = self.model(context_images, context_labels, target_images)
        task_loss = self.loss(logits, target_labels) / self.args.tasks_per_batch
        task_loss.backward()
        accuracy = self.accuracy_fn(logits, target_labels)
        return task_loss, accuracy

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            accuracies = []
            for _ in range(NUM_VALIDATION_TASKS):
                task_dict = self.dataset.get_validation_task(self.args.test_way, self.args.test_shot, self.args.query)
                context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict, shuffle=False)
                logits = self.model(context_images, context_labels, target_images)
                accuracy = self.accuracy_fn(logits, target_labels)
                accuracies.append(accuracy)
                del logits

            accuracy = np.array(accuracies).mean() * 100.0
            confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
            print_and_log(self.logfile, 'Validation Accuracy: {0:3.1f}+/-{1:2.1f}'.format(accuracy, confidence))

        return accuracy

    def test_leave_one_out(self, path):
        print_and_log(self.logfile, "")  # add a blank line
        print_and_log(self.logfile, 'Testing model {0:}: '.format(path))
        self.model = self.init_model()
        self.model.load_state_dict(torch.load(path))

        self.model.eval()
        with torch.no_grad():

            task_dict = self.dataset.get_test_task(self.args.test_way, # task way
                                                   self.args.test_shot, # context set shot
                                                   self.args.query) # target shot
            context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict, shuffle=False)

            logits = self.model(context_images, context_labels, target_images)
            true_accuracy = self.accuracy_fn(logits, target_labels)
            print_and_log(self.logfile, 'True accuracy: {0:3.1f}'.format(true_accuracy))

            import pdb; pdb.set_trace()

            accuracy_loo = []
            for i, im in enumerate(context_images):
                if i == 0:
                    context_images_loo = context_images[i + 1:]
                    context_labels_loo = context_labels[i + 1:]
                else:
                    context_images_loo = context_images[0:i] + context_images[i + 1:]
                    context_labels_loo = context_labels[0:i] + context_labels[i + 1:]

                logits = self.model(context_images_loo, context_labels_loo, target_images)
                accuracy = self.accuracy_fn(logits, target_labels)
                print_and_log(self.logfile, 'Loo {} accuracy: {0:3.1f}'.format(i, true_accuracy - accuracy))
                accuracy_loo.append(accuracy)
                del logits


    def test(self, path):
        print_and_log(self.logfile, "")  # add a blank line
        print_and_log(self.logfile, 'Testing model {0:}: '.format(path))
        self.model = self.init_model()
        self.model.load_state_dict(torch.load(path))

        self.model.eval()
        with torch.no_grad():
            accuracies = []
            for _ in range(NUM_TEST_TASKS):
                task_dict = self.dataset.get_test_task(self.args.test_way,
                                                       self.args.test_shot,
                                                       self.args.query)
                context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict, shuffle=False)
                logits = self.model(context_images, context_labels, target_images)
                accuracy = self.accuracy_fn(logits, target_labels)
                accuracies.append(accuracy)
                del logits

            accuracy = np.array(accuracies).mean() * 100.0
            accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

            print_and_log(self.logfile, 'Test Accuracy: {0:3.1f}+/-{1:2.1f}'.format(accuracy, accuracy_confidence))

    def prepare_task(self, task_dict, shuffle):
        context_images, context_labels = task_dict['context_images'], task_dict['context_labels']
        target_images, target_labels = task_dict['target_images'], task_dict['target_labels']

        if shuffle:
            context_images, context_labels = self.shuffle(context_images, context_labels)
            target_images, target_labels = self.shuffle(target_images, target_labels)

        context_images = context_images.to(self.device)
        target_images = target_images.to(self.device)
        context_labels = torch.from_numpy(context_labels).to(self.device)
        target_labels = torch.from_numpy(target_labels).type(torch.LongTensor).to(self.device)

        return context_images, target_images, context_labels, target_labels

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]

    def save_checkpoint(self, iteration):
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_validation_accuracy,
        }, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'checkpoint.pt'))
        self.start_iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_validation_accuracy = checkpoint['best_accuracy']

    def adjust_learning_rate(self, iteration):
        """
        Sets the learning rate to the initial LR decayed by 2 every 2000 tasks
        """
        lr = self.args.learning_rate * (0.5 ** (iteration // 2000))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


if __name__ == "__main__":
    main()
