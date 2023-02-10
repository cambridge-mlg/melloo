import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from config_networks import ConfigureNetworks
from set_encoder import mean_pooling
from utils import MetaLearningState, extract_class_indices
from attention import DotProdAttention, MultiHeadAttention, CrossTransformer
from einops import rearrange
from torch.linalg import norm


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
        #self.class_representations = OrderedDict()  # Dictionary mapping class label (integer) to encoded representation
        self.context_features = None
        self.target_features = None
        self.classifier_params = None

        if self.args.classifier == "protonets_attention" :
            self.cross_attention = DotProdAttention(temperature=self.args.attention_temperature)

        if self.args.classifier == "protonets_cross_transformer":
            self.cross_attention = CrossTransformer(temperature=self.args.attention_temperature)
            
        #if self.args.classifier == "protonets_mahalanobis":
        #    self.class_precision_matrices = OrderedDict() # Dictionary mapping class label (integer) to regularized precision matrices estimated



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
            self.set_class_prototypes(prototypes)
            logits = self._cross_transformer_euclidiean_distances(prototypes, target_embeddings, h, w)

        elif self.args.classifier == "protonets_mahalanobis":
            logits = self._protonets_mahalanobis_classifier(self.context_features, self.target_features, context_labels)
        else:
            print("Unsupported model type requested")
            return

        return logits

    def _protonets_mahalanobis_classifier(self, context_features, target_features, context_labels):
        """
        SCM: we build both class representations and the regularized covariance estimates.
        """
        # get the class means and covariance estimates in tensor form
        class_means, class_precision_matrices = self._build_class_reps_and_covariance_estimates(context_features, context_labels)
        self.classifier_params = {
            'class_means': class_means,
            'class_precision_matrices': class_precision_matrices
        }

        """
        SCM: calculating the Mahalanobis distance between query examples and the class means
        including the class precision estimates in the calculations, reshaping the distances
        and multiplying by -1 to produce the sample logits
        """
        # grabbing the number of classes and query examples for easier use later in the function
        number_of_classes = class_means.size(0)
        number_of_targets = target_features.size(0)
        
        repeated_target = target_features.repeat(1, number_of_classes).view(-1, class_means.size(1))
        repeated_class_means = class_means.repeat(number_of_targets, 1)
        repeated_difference = (repeated_class_means - repeated_target)
        repeated_difference = repeated_difference.view(number_of_targets, number_of_classes,
                                                       repeated_difference.size(1)).permute(1, 0, 2)
        first_half = torch.matmul(repeated_difference, class_precision_matrices)
        logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1
        return logits
    
    def set_class_prototypes(self, prototypes):
        self.classifier_params = {"class_prototypes": prototypes}
    
    def _build_class_reps_and_covariance_estimates(self, context_features, context_labels):
        """
        Construct and return class level representations and class covariance estimattes for each class in task.
        :param context_features: (torch.tensor) Adapted feature representation for each image in the context set.
        :param context_labels: (torch.tensor) Label for each image in the context set.
        :return: (void) Updates the internal class representation and class covariance estimates dictionary.
        """

        """
        SCM: calculating a task level covariance estimate using the provided function.
        """
        task_covariance_estimate = self.estimate_cov(context_features)
        class_representations = OrderedDict()
        class_precision_matrices = OrderedDict()
        for c in torch.unique(context_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(context_features, 0, extract_class_indices(context_labels, c))
            # mean pooling examples to form class means
            class_rep = mean_pooling(class_features)
            # updating the class representations dictionary with the mean pooled representation
            class_representations[c.item()] = class_rep
            """
            Calculating the mixing ratio lambda_k_tau for regularizing the class level estimate with the task level estimate."
            Then using this ratio, to mix the two estimate; further regularizing with the identity matrix to assure invertability, and then
            inverting the resulting matrix, to obtain the regularized precision matrix. This tensor is then saved in the corresponding
            dictionary for use later in infering of the query data points.
            """
            lambda_k_tau = (class_features.size(0) / (class_features.size(0) + 1))
            class_precision_matrices[c.item()] = torch.inverse(
                (lambda_k_tau * self.estimate_cov(class_features)) + ((1 - lambda_k_tau) * task_covariance_estimate) \
                + torch.eye(class_features.size(1), class_features.size(1)).cuda(0))

        class_means = torch.stack(list(class_representations.values())).squeeze(1)
        class_precs = torch.stack(list(class_precision_matrices.values())) 

        return class_means, class_precs
        
    def estimate_cov(self, examples, rowvar=False, inplace=False):
        """
        SCM: unction based on the suggested implementation of Modar Tensai
        and his answer as noted in: https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5

        Estimate a covariance matrix given data.

        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

        Args:
            examples: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.

        Returns:
            The covariance matrix of the variables.
        """
        if examples.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if examples.dim() < 2:
            examples = examples.view(1, -1)
        if not rowvar and examples.size(0) != 1:
            examples = examples.t()
        factor = 1.0 / (examples.size(1) - 1)
        if inplace:
            examples -= torch.mean(examples, dim=1, keepdim=True)
        else:
            examples = examples - torch.mean(examples, dim=1, keepdim=True)
        examples_t = examples.t()
        return factor * examples.matmul(examples_t).squeeze()

        
    def _protonets_euclidean_classifier(self, context_features, target_features, context_labels):
        prototypes = self._compute_class_prototypes(context_features, context_labels)
        self.set_class_prototypes(prototypes)
        logits = self._euclidean_distances(target_features, prototypes)
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
        # Expand out the target features, so that we have a copy to subtract from each of the prototypes
        # Similarly, expand out the prototypes, but expand along different dimension so that we get 
        # qp1 - proto1, qp1 - proto2, ... qp2 - proto1, qp2 - proto2, ....
        distances = (target_features.unsqueeze(1).expand(num_target_features, num_prototypes, -1) -
                     class_prototypes.unsqueeze(0).expand(num_target_features, num_prototypes, -1)).pow(2).sum(dim=2)

        return -distances

    def _protonets_euclidean_attention_classifier(self, context_features, target_features, context_labels):
        prototypes, attention_weights = self._compute_class_prototypes_with_attention(context_features, target_features,
                                                                         context_labels)
        self.set_class_prototypes(prototypes)
        logits = self._euclidean_distances_with_attention(target_features, prototypes)
        return logits, attention_weights

    def _compute_class_prototypes_with_attention(self, context_features, target_features, context_labels):
        dk = context_features.size(1)
        target_set_size = target_features.size(0)
        class_representations = []
        attention_weights = []
        for c in torch.unique(context_labels):
            # select feature vectors which have class c
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

    def _euclidean_distances_with_attention(self, target_features, class_prototypes):
        num_target_features = target_features.shape[0]
        num_prototypes = class_prototypes.shape[0]
        # Expand out the target features, so that we have a copy to subtract from each of the prototypes
        # The attention-prototypes already have a value per target point, so no need to expan. 
        # Instead, we're going to subtract a target-point-specific version of the prototype from the target point
        # (which was calculated using attention)
        # Just permute so the dimensions match up, then calculate euclidean distance.
        distances = (target_features.unsqueeze(1).expand(num_target_features, num_prototypes, -1) -
                     class_prototypes.permute(1, 0, 2)).pow(2).sum(dim=2)

        return -distances
        
    def _cross_transformer_euclidiean_distances(self, prototypes, target_embeddings, h, w):
        euclidean_dist = ((target_embeddings - prototypes) ** 2).sum(dim = -1) / (h * w)
        return -euclidean_dist

    def classifer_regularization_term(self):
        if "class_prototypes" not in self.classifier_params or self.classifier_params["class_prototypes"] is None:
            print("Classifier has no prototypes saved. Either no forward pass has happened yet, or the regularization term is being calculated out of sync")
            return None
        
        if self.args.classifier == "protonets_cross_transformer":
            print("Though it makes sense to support this, we don't yet")
            return None
        elif self.args.classifier != "protonets_attention" and self.args.classifier != "protonets_euclidean":
            print("Classifier regularization term not available for selected classifer: {}".format(self.args.classifier))
            return None
        
        prototypes = self.classifier_params["class_prototypes"]
        
        # In all cases, the most recently calculated prototypes are saved on the model
        # The classifier head's parameters are wk and bk with wk = 2ck, bk = -ck^Tck
        # where ck denotes the k-th class's prototype
        # So if we want to write this as a matrix, we have W = 2 [c1 c2 ... cK] (with each ck a row vector)
        # so that W has dim (K x embedding_dim) and
        # b = -[c1^Tc1 c2^tc2 ... cK^TcK], having dim (K x 1)
        # Rewriting W f + b again so that we only have one matrix of parameters, we
        # augment W to have an extra column containing b, and f is augmented to have an extra dimension = 1 
        
        normalize_denom = 1.0
        if self.args.classifier == "protonets_attention":
            # permute the embedding so that we have the prototypes per target point
            prototypes = prototypes.permute(1, 0, 2)
            normalize_denom = prototypes.shape[0]
            
        cs_per_target_pt = 2 * prototypes
        bs = -(prototypes * prototypes).sum(dim=-1)
        params = torch.cat((cs_per_target_pt, bs.unsqueeze(-1)), dim=-1)
        
        # Clear the reference to our own prototypes for now, so that if we accidentally call this function out of sync, we will notice        
        self.classifier_params = None
        
        # Calculate the L2 norm of the params, which is either the entire param matrix for protonets_euclidean
        # or over the classifier heads for each target point, in the dot product attention case
        return norm(params, dim=(-2, -1)).sum()/float(normalize_denom)

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

    #def _get_classifier_params(self):
    #    # This is None for the protonets_attention and protonets_euclidean
    #    classifier_params = self.classifier_adaptation_network(self.class_representations)
    #    return classifier_params

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

    

    def get_feature_extractor_params(self):
        return self.feature_extractor_params

    def get_classifier_params(self):
        return self.classifier_params
