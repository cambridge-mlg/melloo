import torch
import torch.nn as nn
from collections import OrderedDict
from utils import split_first_dim_linear
from config_networks import ConfigureNetworks
from set_encoder import mean_pooling

NUM_SAMPLES=1


class Art_Wrapper(nn.Module):
    """
    Wrapper for the main CNAPs model.
    :param device: (str) Device (gpu or cpu) on which model resides.
    :param use_two_gpus: (bool) Whether to paralleize the model (model parallelism) across two GPUs.
    :param args: (Argparser) Arparse object containing model hyper-parameters.
    """
    def __init__(self, cnaps_model):
        super(Art_Wrapper, self).__init__()
        self.cnaps_model = cnaps_model

    #Don't we also need the adv_context_image's label?
    def init_data(self, context_images, context_labels, target_images, adv_context_index):
        self.context_images = context_images
        self.context_labels = context_labels
        self.target_images = target_images
        self.adv_context_index = adv_context_index

    def forward(self, context_image_adv):
        """
        Forward pass through the model for one episode.
        :param context_images: (torch.tensor) Images in the context set (batch x C x H x W).
        :param context_labels: (torch.tensor) Labels for the context set (batch x 1 -- integer representation).
        :param target_images: (torch.tensor) Images in the target set (batch x C x H x W).
        :return: (torch.tensor) Categorical distribution on label set for each image in target set (batch x num_labels).
        """

        context_images_attack = torch.cat([self.context_images[0:self.adv_context_index], context_image_adv,
            self.context_images[self.adv_context_index + 1:]], dim=0)
        logits = self.cnaps_model(context_images_attack, self.context_labels, self.target_images)
        return torch.unsqueeze(logits[0][0], dim=0)



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
