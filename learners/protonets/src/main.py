import torch
import numpy as np
import argparse
import os
from PIL import Image
from learners.protonets.src.utils import print_and_log, get_log_files, categorical_accuracy, loss
from learners.protonets.src.model import ProtoNets
from learners.protonets.src.data import MiniImageNetData, OmniglotData

NUM_VALIDATION_TASKS = 400
NUM_TEST_TASKS = 1
PRINT_FREQUENCY = 100


def convert_to_image(image_array, scaling='neg_one_to_one'):
    image_array = image_array.transpose([1, 2, 0])
    mode = 'RGB'
    if image_array.shape[2] == 1:  # single channel image
        image_array = image_array.squeeze()
        mode = 'L'
    if scaling == 'neg_one_to_one':
        im = Image.fromarray(np.clip((image_array + 1.0) * 127.5 + 0.5, 0, 255).astype(np.uint8), mode=mode)
    else:
        im = Image.fromarray(np.clip(image_array * 255.0, 0, 255).astype(np.uint8), mode=mode)
    return im


def save_image(image_array, save_path, scaling='neg_one_to_one'):
    im = convert_to_image(image_array, scaling)
    im.save(save_path)


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
        parser.add_argument("--exp_type", choices=["test", "loo", "compress", "outlier"], default="loo",
                            help="Which experiment to run. These are only valid in test mode, since that's when our experiments are valid")

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
            if self.args.exp_type == 'test':
                self.test(self.args.test_model_path)
            elif self.args.exp_type == 'loo':
                self.test_leave_one_out(self.args.test_model_path)
            elif self.args.exp_type == 'compress':
                self.compress_task(self.args.test_model_path)

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

    def compress_task(self, path):
        print_and_log(self.logfile, "")  # add a blank line
        print_and_log(self.logfile, 'Compression experiment on model {0:}: '.format(path))
        self.model = self.init_model()
        self.model.load_state_dict(torch.load(path))

        self.model.eval()
        with torch.no_grad():

            task_dict = self.dataset.get_test_task(self.args.test_way, # task way
                                                   self.args.test_shot, # context set shot
                                                   self.args.query) # target shot
            context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict, shuffle=False)

            for i in range(len(context_images)):
                save_image(context_images[i].cpu().detach().numpy(),
                           os.path.join(self.checkpoint_dir, 'context_{}.png'.format(i)),
                           scaling='neg_one_to_one')

            logits = self.model(context_images, context_labels, target_images)
			# THINK: Should we re-calculate the true accuracy for each omission?
            true_accuracy = self.accuracy_fn(logits, target_labels)
            print_and_log(self.logfile, 'True accuracy: {0:3.1f}'.format(true_accuracy))
            
            # We should leave at least one example per class. So max omissions is (total images) - (one per class)
            max_omissions = context_images.shape[0] - self.args.test_way
            acc_overall = np.zeros(max_omissions)
            acc_per_class = np.zeros((self.args.test_way, max_omissions))
            
            # Systematically omit context points until we have only one per class
            for k in range(max_omissions):
				acc_loo_k, acc_per_class_k = loo(context_images, context_labels, target_images, target_labels)
				
				# THINK: Should we select based on acc per class or overall? Let's start with overall.
				most_unhelpful_index = -1
				most_unhelpful_effect =  201.0 # Large number (outside max possible range of effect)
				
				for i, acc in enumerate(acc_loo_k):
					effect = (true_accuracy - acc) * 100
					# If effect > 0, then dropping the point hurt accuracy.
					# Conversely, if effect < 0, then dropping the point helped accuracy
					if effect < most_unhelpful_effect or most_unhelpful_index == -1:
						# THINK: We need to do something if the most unhelpful point is somehow also the last point 
						# in a particular class.
						most_unhelpful_effect = effect
						most_unhelpful_index = i
						
				
				print_and_log(self.logfile, 
					'\tDropped point #{0:} Index {1:} (Class {2:})  accuracy: {3:3.5f}'.format(k+1,
					most_unhelpful_index, context_labels[most_unhelpful_index] most_unhelpful_effect))
				
				# Drop the most unhelpful point and repeat.
				if most_unhelpful_index == 0:
					context_images = context_images[most_unhelpful_index + 1:]
					context_labels = context_labels[most_unhelpful_index + 1:]
				else:
					context_images = torch.cat((context_images[0:most_unhelpful_index], context_images[most_unhelpful_index + 1:]), 0)
					context_labels = torch.cat((context_labels[0:most_unhelpful_index], context_labels[most_unhelpful_index + 1:]), 0)
					
				num_unique = torch.unique(context_labels_loo).shape[0]
				if num_unique < self.args.test_way:
					print_and_log(self.logfile, "\tMost unhelpful point was last of class. ")
					
					
				logits = self.model(context_images, context_labels, target_images)
				accuracy = self.accuracy_fn(logits, target_labels)
				acc_overall[k] = accuracy
				for c in range(0, self.args.test_way):
					target_images_c = target_images[c*(self.args.query): (c+1)*self.args.query]
					target_labels_c = target_labels[c*(self.args.query): (c+1)*self.args.query]
					logits = self.model(context_images, context_labels, target_images_c)
					accuracy = self.accuracy_fn(logits, target_labels_c)
					acc_per_class[c][k] = accuracy
				del logits


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

            for i in range(len(context_images)):
                save_image(context_images[i].cpu().detach().numpy(),
                           os.path.join(self.checkpoint_dir, 'context_{}.png'.format(i)),
                           scaling='neg_one_to_one')

            logits = self.model(context_images, context_labels, target_images)
            true_accuracy = self.accuracy_fn(logits, target_labels)
            print_and_log(self.logfile, 'True accuracy: {0:3.1f}'.format(true_accuracy))
            
            accuracy_loo, accuracy_per_class = loo(context_images, context_labels, target_images, target_labels)
			
			# Log results
            for i in range(context_images.shape[0]): 
                print_and_log(self.logfile, 'Loo {0:} accuracy: {1:3.5f}'.format(i, true_accuracy - accuracy_loo[i]))
                for c in range(0, self.args.test_way):
                    print_and_log(self.logfile, '\tLoo {0:} class {1:}  accuracy: {2:3.5f}'.format(i, c, true_accuracy - accuracy_per_class[c][i]))
                    
            
    # Loo helper function, returns the overall accuracies as well as per class accuracies when leaving out 
    # each of the context points in turn
    def loo(self, context_images, context_labels, target_images, target_labels):
        # Initialize return values
		accuracy_loo = np.zeros(context_images.shape[0])
        accuracy_per_class = np.zeros(self.args.test_way, context_images.shape[0]) 
		
		# Not sure where on the model to grab the loss from.
		# loss_loo = np.zeros(context_images.shape[0])
		# loss_per_class = np.zeros(self.args.test_way, context_images.shape[0])
		
        for i, im in enumerate(context_images):
            if i == 0:
                context_images_loo = context_images[i + 1:]
                context_labels_loo = context_labels[i + 1:]
            else:
                context_images_loo = torch.cat((context_images[0:i], context_images[i + 1:]), 0)
                context_labels_loo = torch.cat((context_labels[0:i], context_labels[i + 1:]), 0)
                
            # Check that we leave at least one image per class. We can do this by counting the number of unique labels in the
            # loo context labels and checking they're equal to the way
            num_unique = torch.unique(context_labels_loo).shape[0]
            if num_unique < self.args.test_way:
                # Skip this image, it doesn't make sense to leave out all a class's images
                accuracy_loo[i] = np.nan
                for c in range(0, self.args.test_way):
                    accuracy_per_class[c][i] = np.nan
				continue

            logits = self.model(context_images_loo, context_labels_loo, target_images)
            accuracy = self.accuracy_fn(logits, target_labels)
            accuracy_loo[i] = accuracy
            for c in range(0, self.args.test_way):
                target_images_c = target_images[c*(self.args.query): (c+1)*self.args.query]
                target_labels_c = target_labels[c*(self.args.query): (c+1)*self.args.query]
                logits = self.model(context_images_loo, context_labels_loo, target_images_c)
                accuracy = self.accuracy_fn(logits, target_labels_c)
                accuracy_per_class[c][k] = accuracy
				del logits
        
        return accuracy_loo, accuracy_per_class #, loss_loo, loss_per_class

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
