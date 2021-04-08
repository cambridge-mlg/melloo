from resnet import film_resnet18, resnet18
from adaptation_networks import NullFeatureAdaptationNetwork, FilmAdaptationNetwork, \
    LinearClassifierAdaptationNetwork, FilmLayerNetwork, AttentiveLinearClassifierAdaptationNetwork, \
    CrossTransformerClassifierAdaptationNetwork
from set_encoder import SetEncoder
from utils import linear_classifier, attentive_linear_classifier, cross_transformer_linear_classifier


"""
Creates the set encoder, feature extractor, feature adaptation, classifier, and classifier adaptation networks.
"""


class ConfigureNetworks:
    def __init__(self, pretrained_resnet_path, feature_adaptation, batch_normalization, classifier,
                 do_not_freeze_feature_extractor):

        self.encoder = SetEncoder(batch_normalization=batch_normalization)
        z_g_dim = self.encoder.pre_pooling_fn.output_size
        skip_pooling = (classifier == "protonets_cross_transformer") or\
                       (classifier == "versa_cross_transformer")

        # parameters for ResNet18
        num_maps_per_layer = [64, 128, 256, 512]
        num_blocks_per_layer = [2, 2, 2, 2]

        if feature_adaptation == "no_adaptation":
            self.feature_extractor = resnet18(
                pretrained=True,
                pretrained_model_path=pretrained_resnet_path,
                batch_normalization=batch_normalization,
                skip_pooling=skip_pooling
            )

            self.feature_adaptation_network = NullFeatureAdaptationNetwork()

        elif feature_adaptation == "film":
            self.feature_extractor = film_resnet18(
                pretrained=True,
                pretrained_model_path=pretrained_resnet_path,
                batch_normalization=batch_normalization,
                skip_pooling=skip_pooling
            )
            self.feature_adaptation_network = FilmAdaptationNetwork(
                layer=FilmLayerNetwork,
                num_maps_per_layer=num_maps_per_layer,
                num_blocks_per_layer=num_blocks_per_layer,
                z_g_dim=z_g_dim
            )

        # Freeze the parameters of the feature extractor
        if not do_not_freeze_feature_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # configure the classifier
        if classifier == 'versa':
            self.classifier_adaptation_network = LinearClassifierAdaptationNetwork(self.feature_extractor.output_size)
            self.classifier = linear_classifier
        elif classifier == 'versa_attention':
            self.classifier_adaptation_network = AttentiveLinearClassifierAdaptationNetwork(self.feature_extractor.output_size)
            self.classifier = attentive_linear_classifier
        elif classifier == 'versa_cross_transformer':
            self.classifier_adaptation_network = CrossTransformerClassifierAdaptationNetwork(128)  # put constant in different place
            self.classifier = cross_transformer_linear_classifier
        elif classifier == 'protonets_euclidean' or\
             classifier == 'protonets_mahalanobis' or\
             classifier == 'protonets_kl' or \
             classifier == 'protonets_attention' or \
             classifier == 'protonets_cross_transformer':
            self.classifier_adaptation_network = None
            self.classifier = None

    def get_encoder(self):
        return self.encoder

    def get_classifier(self):
        return self.classifier

    def get_classifier_adaptation(self):
        return self.classifier_adaptation_network

    def get_feature_adaptation(self):
        return self.feature_adaptation_network

    def get_feature_extractor(self):
        return self.feature_extractor
