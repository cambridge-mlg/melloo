import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from config_networks import ConfigureNetworks
from set_encoder import mean_pooling
from utils import MetaLearningState, extract_class_indices
from attention import DotProdAttention, MultiHeadAttention, CrossTransformer
from einops import rearrange


class FewShotClassifier(nn.Module):
    """
    Main model class. Implements several CNAPs models (with / without feature adaptation, with /without auto-regressive
    adaptation parameters generation.
    :param device: (str) Device (gpu or cpu) on which model resides.
    :param use_two_gpus: (bool) Whether to paralleize the model (model parallelism) across two GPUs.
    :param args: (Argparser) Arparse object containing model hyper-parameters.
    """
    def __init__(self, device, use_two_gpus, args):
        super(FewShotClassifier, self).__init__()
        self.args = args
        self.device = device
        self.use_two_gpus = use_two_gpus
        networks = ConfigureNetworks(pretrained_resnet_path=self.args.pretrained_resnet_path,
                                     feature_adaptation=self.args.feature_adaptation,
                                     batch_normalization=args.batch_normalization,
                                     classifier=args.classifier,
                                     do_not_freeze_feature_extractor=args.do_not_freeze_feature_extractor)
        self.set_encoder = networks.get_encoder()
        self.classifier_adaptation_network = networks.get_classifier_adaptation()
        self.classifier = networks.get_classifier()
        self.feature_extractor = networks.get_feature_extractor()
        self.feature_adaptation_network = networks.get_feature_adaptation()
        self.task_representation = None
        self.feature_extractor_params = {}
        self.class_representations = OrderedDict()  # Dictionary mapping class label (integer) to encoded representation
        self.context_features = None
        self.target_features = None

        if self.args.classifier == "protonets_attention" :
            self.cross_attention = DotProdAttention(temperature=self.args.attention_temperature)

        if self.args.classifier == "protonets_cross_transformer":
            self.cross_attention = CrossTransformer(temperature=self.args.attention_temperature)


    def forward(self, context_images, context_labels, target_images, target_labels, meta_learning_state):
        """
        Forward pass through the model for one episode.
        :param context_images: (torch.tensor) Images in the context set (batch x C x H x W).
        :param context_labels: (torch.tensor) Labels for the context set (batch x 1 -- integer representation).
        :param target_images: (torch.tensor) Images in the target set (batch x C x H x W).
        :return: (torch.tensor) Categorical distribution on label set for each image in target set (batch x num_labels).
        """
        # extract context and target features
        if self.args.feature_adaptation != "no_adaptation":
            self.task_representation = self.set_encoder(context_images)
        self.context_features, self.target_features = self._get_features(context_images, target_images, meta_learning_state)

        # classify
        if self.args.classifier == "protonets_euclidean":
            logits = self._protonets_euclidean_classifier(self.context_features, self.target_features, context_labels)

        elif self.args.classifier == "protonets_attention":
            logits, attention_weights = self._protonets_euclidean_attention_classifier(self.context_features, self.target_features,
                                                                    context_labels)
            self.attention_weights = attention_weights

        elif self.args.classifier == "protonets_cross_transformer":
            prototypes, target_embeddings, h, w = self.cross_attention(self.context_features, self.target_features, context_labels)
            logits = self._cross_transformer_euclidiean_distances(prototypes, target_embeddings, h, w)

        else:
            print("Unsupported model type requested")
            return

        return logits

    def get_context_features(self):
        return self.context_features

    def get_target_features(self):
        return self.target_features

    def _get_features(self, context_images, target_images, meta_learning_state):
        """
        Helper function to extract task-dependent feature representation for each image in both context and target sets.
        :param context_images: (torch.tensor) Images in the context set (batch x C x H x W).
        :param target_images: (torch.tensor) Images in the target set (batch x C x H x W).
        :return: (tuple::torch.tensor) Feature representation for each set of images.
        """
        # Parallelize forward pass across multiple GPUs (model parallelism)
        if self.use_two_gpus:
            context_images_1 = context_images.cuda(1)
            target_images_1 = target_images.cuda(1)
            if self.task_representation is not None:
                task_representation_1 = self.task_representation.cuda(1)
                # Get adaptation params by passing context set through the adaptation networks
                self.feature_extractor_params = self.feature_adaptation_network(task_representation_1)

            # Given adaptation parameters for task, conditional forward pass through the adapted feature extractor
            self._set_batch_norm_mode(True, meta_learning_state)
            context_features_1 = self.feature_extractor(context_images_1, self.feature_extractor_params)
            context_features = context_features_1.cuda(0)
            self._set_batch_norm_mode(False, meta_learning_state)
            target_features_1 = self.feature_extractor(target_images_1, self.feature_extractor_params)
            target_features = target_features_1.cuda(0)
        else:
            # Get adaptation params by passing context set through the adaptation networks
            if self.task_representation is not None:
                self.feature_extractor_params = self.feature_adaptation_network(self.task_representation)

            # Given adaptation parameters for task, conditional forward pass through the adapted feature extractor
            self._set_batch_norm_mode(True, meta_learning_state)
            context_features = self.feature_extractor(context_images, self.feature_extractor_params)
            self._set_batch_norm_mode(False, meta_learning_state)
            target_features = self.feature_extractor(target_images, self.feature_extractor_params)

        return context_features, target_features

    def _get_classifier_params(self):
        classifier_params = self.classifier_adaptation_network(self.class_representations)
        return classifier_params

    def distribute_model(self):
        self.feature_extractor.cuda(1)
        self.feature_adaptation_network.cuda(1)

    def _set_batch_norm_mode(self, context, meta_learning_state):
        if self.args.batch_normalization == "basic":
            self.feature_extractor.eval()  # ignore context and state flag
        elif self.args.batch_normalization == "standard":  # ignore context flag and use state
            if meta_learning_state == MetaLearningState.META_TRAIN:
                self.feature_extractor.train()
            else:
                self.feature_extractor.eval()
        else:  # task_norm-i
            # respect context flag, regardless of state
            if context:
                self.feature_extractor.train()
            else:
                self.feature_extractor.eval()

    def _protonets_euclidean_classifier(self, context_features, target_features, context_labels):
        class_prototypes = self._compute_class_prototypes(context_features, context_labels)
        logits = self._euclidean_distances(target_features, class_prototypes)
        return logits

    def _compute_class_prototypes(self, context_features, context_labels):
        means = []
        for c in torch.unique(context_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(context_features, 0, extract_class_indices(context_labels, c))
            means.append(torch.mean(class_features, dim=0))
        return torch.stack(means)

    def _euclidean_distances(self, target_features, class_prototypes):
        num_target_features = target_features.shape[0]
        num_prototypes = class_prototypes.shape[0]

        distances = (target_features.unsqueeze(1).expand(num_target_features, num_prototypes, -1) -
                     class_prototypes.unsqueeze(0).expand(num_target_features, num_prototypes, -1)).pow(2).sum(dim=2)

        return -distances

    def _cross_transformer_euclidiean_distances(self, prototypes, target_embeddings, h, w):
        euclidean_dist = ((target_embeddings - prototypes) ** 2).sum(dim = -1) / (h * w)
        return -euclidean_dist

    def _protonets_euclidean_attention_classifier(self, context_features, target_features, context_labels):
        class_prototypes, attention_weights = self._compute_class_prototypes_with_attention(context_features, target_features,
                                                                         context_labels)
        logits = self._euclidean_distances_with_attention(target_features, class_prototypes)
        return logits, attention_weights
        
    def _euclidean_distances_with_attention(self, target_features, class_prototypes):
        num_target_features = target_features.shape[0]
        num_prototypes = class_prototypes.shape[0]

        distances = (target_features.unsqueeze(1).expand(num_target_features, num_prototypes, -1) -
                     class_prototypes.permute(1, 0, 2)).pow(2).sum(dim=2)

        return -distances

    def _compute_class_prototypes_with_attention(self, context_features, target_features, context_labels):
        dk = context_features.size(1)
        target_set_size = target_features.size(0)
        class_representations = []
        attention_weights = []
        for c in torch.unique(context_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(context_features, 0, extract_class_indices(context_labels, c))

            # replicate the class features x target_set_size
            class_keys = torch.unsqueeze(class_features, dim=0)
            class_keys = class_keys.repeat(target_set_size, 1, 1)

            # unsqueeze targets at dim = 1
            class_queries = target_features.unsqueeze(1)

            class_prototype, attn_weights = self.cross_attention(keys=class_keys, queries=class_queries, values=class_keys)
    
            class_representations.append(class_prototype.squeeze())
            attention_weights.append(attn_weights)

        return torch.stack(class_representations), attention_weights

    def get_feature_extractor_params(self):
        return self.feature_extractor_params

    def get_classifier_params(self):
        return self.classifier_params
