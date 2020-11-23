import torch
import torch.nn as nn
from learners.protonets.src.feature_extractor import ConvNet
from learners.protonets.src.utils import euclidean_metric


class ProtoNets(nn.Module):
    """
    Main model class.
    :param args: (Argparser) Arparse object containing model hyper-parameters.
    """
    def __init__(self, args):
        super(ProtoNets, self).__init__()
        self.args = args
        if self.args.dataset == 'omniglot':
            in_channels = 1
        else:
            in_channels = 3

        self.feature_extractor = ConvNet(in_channels=in_channels)

    def forward(self, context_images, context_labels, target_images):
        """
        Forward pass through the model for one episode.
        :param context_images: (torch.tensor) Images in the context set (batch x C x H x W).
        :param context_labels: (torch.tensor) Labels for the context set (batch x 1 -- integer representation).
        :param target_images: (torch.tensor) Images in the target set (batch x C x H x W).
        :return: (torch.tensor) Categorical distribution on label set for each image in target set (batch x num_labels).
        """

        context_features = self.feature_extractor(context_images)
        if self.training:
            way = self.args.train_way
        else:
            way = self.args.test_way
        prototypes = self._compute_prototypes(context_features, context_labels, way)

        target_features = self.feature_extractor(target_images)
        logits = euclidean_metric(target_features, prototypes)
        return logits

    def _compute_prototypes(self, context_features, context_labels, way):
        prototypes = []
        for c in torch.unique(context_labels):
            class_features = torch.index_select(context_features, 0, self._extract_class_indices(context_labels, c))
            prototypes.append(torch.mean(class_features, dim=0, keepdim=True))

        return torch.squeeze(torch.stack(prototypes))

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector
