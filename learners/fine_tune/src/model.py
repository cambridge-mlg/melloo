import torch
import numpy as np
from utils import loss_fn, accuracy
from classifier import FilmClassifier
from feature_extractor import create_feature_extractor


class FineTuner:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.classifier = None
        self.feature_extractor = create_feature_extractor(
            feature_extractor_family=self.args.feature_extractor,
            feature_adaptation=self.args.feature_adaptation,
            pretrained_path=self.args.pretrained_feature_extractor_path
        ).to(device)
        self.loss = loss_fn
        self.accuracy = accuracy

    
    def zero_grad(self):
        #Not entirely sure what to do here. For now do nothing?
        return

    def forward(self, context_images, context_labels, target_images):
        self.fine_tune(context_images, context_labels)
        return self.test_linear(target_images, logits=True)

    def fine_tune(self, context_images, context_labels):
        self.classifier = FilmClassifier(
            num_classes=len(torch.unique(context_labels)),
            feature_extractor=self.feature_extractor,
            feature_adaptation=self.args.feature_adaptation,
            feature_extractor_family=self.args.feature_extractor
        ).to(self.device)
        self.optimizer = torch.optim.SGD(
            self.classifier.parameters(),
            lr=self.args.learning_rate,
            momentum=0.9,
            weight_decay=self.args.weight_decay)
        self.classifier.feature_extractor.eval()
        torch.set_grad_enabled(True)
        context_set_size = len(context_labels)
        num_batches = int(np.ceil(float(context_set_size) / float(self.args.batch_size)))
        #self.optimizer.zero_grad()
        for iteration in range(self.args.iterations):
            self.adjust_learning_rate(iteration)
            for batch in range(num_batches):
                batch_start_index, batch_end_index = self._get_batch_indices(batch, context_set_size)
                logits = self.classifier(context_images[batch_start_index : batch_end_index])
                loss = self.loss(logits, context_labels[batch_start_index : batch_end_index])
                regularization_term = self.classifier.film_adapter.regularization_term()
                loss += self.args.regularizer_scaling * regularization_term
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                del logits
                # print("i={}, loss={}".format(iteration, loss))

    def _get_batch_indices(self, index, last_element):
        batch_start_index = index * self.args.batch_size
        batch_end_index = batch_start_index + self.args.batch_size
        if batch_end_index > last_element:
            batch_end_index = last_element
        return batch_start_index, batch_end_index

    def test_linear(self, images, labels=None, logits=False):
        self.classifier.feature_extractor.eval()
        test_set_size = len(images)
        
        # with torch.no_grad():
        if not logits:
            assert labels is not None
            num_batches = int(np.ceil(float(test_set_size) / float(self.args.batch_size)))
            accuracies = []
            for batch in range(num_batches):
                batch_start_index, batch_end_index = self._get_batch_indices(batch, test_set_size)
                logits = self.classifier(images[batch_start_index : batch_end_index])
                accuracy = self.accuracy(logits, labels[batch_start_index : batch_end_index])
                del logits
                accuracies.append(accuracy.item())
            return np.array(accuracies).mean()
        else:
            return self.classifier(images)
                        
                    
    def adjust_learning_rate(self, iteration):
        if iteration > 66:
            lr = 0.01
        elif iteration > 33:
            lr = 0.025
        else:
            lr = self.args.learning_rate

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

