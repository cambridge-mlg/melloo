import torch.nn as nn
from learners.protonets.src.feature_extractor import ConvNet, BottleneckNet
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
            shot = self.args.train_shot
            way = self.args.train_way
        else:
            shot = self.args.test_shot
            way = self.args.test_way
        prototypes = context_features.reshape(way, shot, -1).mean(dim=1)

        target_features = self.feature_extractor(target_images)
        logits = euclidean_metric(target_features, prototypes)
        return logits

    def forward_embeddings(self, context_features, target_features):
        """
        Forward pass through the model for one episode.
        :param context_images: (torch.tensor) Images in the context set (batch x C x H x W).
        :param context_labels: (torch.tensor) Labels for the context set (batch x 1 -- integer representation).
        :param target_images: (torch.tensor) Images in the target set (batch x C x H x W).
        :return: (torch.tensor) Categorical distribution on label set for each image in target set (batch x num_labels).
        """

        if self.training:
            shot = self.args.train_shot
            way = self.args.train_way
        else:
            shot = self.args.test_shot
            way = self.args.test_way
        prototypes = context_features.reshape(way, shot, -1).mean(dim=1)

        logits = euclidean_metric(target_features, prototypes)
        return logits
