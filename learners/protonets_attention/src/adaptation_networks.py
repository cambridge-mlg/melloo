import torch
import torch.nn as nn
from einops import rearrange


class DenseResidualLayer(nn.Module):
    """
    PyTorch like layer for standard linear layer with identity residual connection.
    :param num_features: (int) Number of input / output units for the layer.
    """
    def __init__(self, num_features):
        super(DenseResidualLayer, self).__init__()
        self.linear = nn.Linear(num_features, num_features)

    def forward(self, x):
        """
        Forward-pass through the layer. Implements the following computation:

                f(x) = f_theta(x) + x
                f_theta(x) = W^T x + b

        :param x: (torch.tensor) Input representation to apply layer to ( dim(x) = (batch, num_features) ).
        :return: (torch.tensor) Return f(x) ( dim(f(x) = (batch, num_features) ).
        """
        identity = x
        out = self.linear(x)
        out += identity
        return out


class DenseResidualBlock(nn.Module):
    """
    Wrapping a number of residual layers for residual block. Will be used as building block in FiLM hyper-networks.
    :param in_size: (int) Number of features for input representation.
    :param out_size: (int) Number of features for output representation.
    """
    def __init__(self, in_size, out_size):
        super(DenseResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_size, out_size)
        self.linear2 = nn.Linear(out_size, out_size)
        self.linear3 = nn.Linear(out_size, out_size)
        self.elu = nn.ELU()

    def forward(self, x):
        """
        Forward pass through residual block. Implements following computation:

                h = f3( f2( f1(x) ) ) + x
                or
                h = f3( f2( f1(x) ) )

                where fi(x) = Elu( Wi^T x + bi )

        :param x: (torch.tensor) Input representation to apply layer to ( dim(x) = (batch, in_size) ).
        :return: (torch.tensor) Return f(x) ( dim(f(x) = (batch, out_size) ).
        """
        identity = x
        out = self.linear1(x)
        out = self.elu(out)
        out = self.linear2(out)
        out = self.elu(out)
        out = self.linear3(out)
        if x.shape[-1] == out.shape[-1]:
            out += identity
        return out


class FilmAdaptationNetwork(nn.Module):
    """
    FiLM adaptation network (outputs FiLM adaptation parameters for all layers in a base feature extractor).
    :param layer: (FilmLayerNetwork) Layer object to be used for adaptation.
    :param num_maps_per_layer: (list::int) Number of feature maps for each layer in the network.
    :param num_blocks_per_layer: (list::int) Number of residual blocks in each layer in the network
                                 (see ResNet file for details about ResNet architectures).
    :param z_g_dim: (int) Dimensionality of network input. For this network, z is shared across all layers.
    """
    def __init__(self, layer, num_maps_per_layer, num_blocks_per_layer, z_g_dim):
        super().__init__()
        self.z_g_dim = z_g_dim
        self.num_maps = num_maps_per_layer
        self.num_blocks = num_blocks_per_layer
        self.num_target_layers = len(self.num_maps)
        self.layer = layer
        self.layers = self.get_layers()

    def get_layers(self):
        """
        Loop over layers of base network and initialize adaptation network.
        :return: (nn.ModuleList) ModuleList containing the adaptation network for each layer in base network.
        """
        layers = nn.ModuleList()
        for num_maps, num_blocks in zip(self.num_maps, self.num_blocks):
            layers.append(
                self.layer(
                    num_maps=num_maps,
                    num_blocks=num_blocks,
                    z_g_dim=self.z_g_dim
                )
            )
        return layers

    def forward(self, x):
        """
        Forward pass through adaptation network to create list of adaptation parameters.
        :param x: (torch.tensor) (z -- task level representation for generating adaptation).
        :return: (list::adaptation_params) Returns a list of adaptation dictionaries, one for each layer in base net.
        """
        return [self.layers[layer](x) for layer in range(self.num_target_layers)]

    def regularization_term(self):
        """
        Simple function to aggregate the regularization terms from each of the layers in the adaptation network.
        :return: (torch.scalar) A order-0 torch tensor with the regularization term for the adaptation net params.
        """
        l2_term = 0
        for layer in self.layers:
            l2_term += layer.regularization_term()
        return l2_term


class FilmLayerNetwork(nn.Module):
    """
    Single adaptation network for generating the parameters of each layer in the base network. Will be wrapped around
    by FilmAdaptationNetwork.
    :param num_maps: (int) Number of output maps to be adapted in base network layer.
    :param num_blocks: (int) Number of blocks being adapted in the base network layer.
    :param z_g_dim: (int) Dimensionality of input to network (task level representation).
    """
    def __init__(self, num_maps, num_blocks, z_g_dim):
        super().__init__()
        self.z_g_dim = z_g_dim
        self.num_maps = num_maps
        self.num_blocks = num_blocks

        # Initialize a simple shared layer for all parameter adapters (gammas and betas)
        self.shared_layer = nn.Sequential(
            nn.Linear(self.z_g_dim, self.num_maps),
            nn.ReLU()
        )

        # Initialize the processors (adaptation networks) and regularization lists for each of the output params
        self.gamma1_processors, self.gamma1_regularizers = torch.nn.ModuleList(), torch.nn.ParameterList()
        self.gamma2_processors, self.gamma2_regularizers = torch.nn.ModuleList(), torch.nn.ParameterList()
        self.beta1_processors, self.beta1_regularizers = torch.nn.ModuleList(), torch.nn.ParameterList()
        self.beta2_processors, self.beta2_regularizers = torch.nn.ModuleList(), torch.nn.ParameterList()

        # Generate the required layers / regularization parameters, and collect them in ModuleLists and ParameterLists
        for _ in range(self.num_blocks):
            self.gamma1_processors.append(self._make_layer(num_maps))
            self.gamma1_regularizers.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(num_maps), 0, 0.001),
                                                               requires_grad=True))

            self.beta1_processors.append(self._make_layer(num_maps))
            self.beta1_regularizers.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(num_maps), 0, 0.001),
                                                              requires_grad=True))

            self.gamma2_processors.append(self._make_layer(num_maps))
            self.gamma2_regularizers.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(num_maps), 0, 0.001),
                                                               requires_grad=True))

            self.beta2_processors.append(self._make_layer(num_maps))
            self.beta2_regularizers.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(num_maps), 0, 0.001),
                                                              requires_grad=True))

    @staticmethod
    def _make_layer(size):
        """
        Simple layer generation method for adaptation network of one of the parameter sets (all have same structure).
        :param size: (int) Number of parameters in layer.
        :return: (nn.Sequential) Three layer dense residual network to generate adaptation parameters.
        """
        return nn.Sequential(
            DenseResidualLayer(size),
            nn.ReLU(),
            DenseResidualLayer(size),
            nn.ReLU(),
            DenseResidualLayer(size)
        )

    def forward(self, x):
        """
        Forward pass through adaptation network.
        :param x: (torch.tensor) Input representation to network (task level representation z).
        :return: (list::dictionaries) Dictionary for every block in layer. Dictionary contains all the parameters
                 necessary to adapt layer in base network. Base network is aware of dict structure and can pull params
                 out during forward pass.
        """
        x = self.shared_layer(x)
        block_params = []
        for block in range(self.num_blocks):
            block_param_dict = {
                'gamma1': self.gamma1_processors[block](x).squeeze() * self.gamma1_regularizers[block] +
                          torch.ones_like(self.gamma1_regularizers[block]),
                'beta1': self.beta1_processors[block](x).squeeze() * self.beta1_regularizers[block],
                'gamma2': self.gamma2_processors[block](x).squeeze() * self.gamma2_regularizers[block] +
                          torch.ones_like(self.gamma2_regularizers[block]),
                'beta2': self.beta2_processors[block](x).squeeze() * self.beta2_regularizers[block]
            }
            block_params.append(block_param_dict)
        return block_params

    def regularization_term(self):
        """
        Compute the regularization term for the parameters. Recall, FiLM applies gamma * x + beta. As such, params
        gamma and beta are regularized to unity, i.e. ||gamma - 1||_2 and ||beta||_2.
        :return: (torch.tensor) Scalar for l2 norm for all parameters according to regularization scheme.
        """
        l2_term = 0
        for gamma_regularizer, beta_regularizer in zip(self.gamma1_regularizers, self.beta1_regularizers):
            l2_term += (gamma_regularizer ** 2).sum()
            l2_term += (beta_regularizer ** 2).sum()
        for gamma_regularizer, beta_regularizer in zip(self.gamma2_regularizers, self.beta2_regularizers):
            l2_term += (gamma_regularizer ** 2).sum()
            l2_term += (beta_regularizer ** 2).sum()
        return l2_term


class NullFeatureAdaptationNetwork(nn.Module):
    """
    Dummy adaptation network for the case of "no_adaptation".
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return {}

    @staticmethod
    def regularization_term():
        return 0


class LinearClassifierAdaptationNetwork(nn.Module):
    """
    Versa-style adaptation network for linear classifier (see https://arxiv.org/abs/1805.09921 for full details).
    :param d_theta: (int) Input / output feature dimensionality for layer.
    """
    def __init__(self, d_theta):
        super(LinearClassifierAdaptationNetwork, self).__init__()
        self.weight_means_processor = self._make_mean_dense_block(d_theta, d_theta)
        self.bias_means_processor = self._make_mean_dense_block(d_theta, 1)

    @staticmethod
    def _make_mean_dense_block(in_size, out_size):
        """
        Simple method for generating different types of blocks. Final code only uses dense residual blocks.
        :param in_size: (int) Input representation dimensionality.
        :param out_size: (int) Output representation dimensionality.
        :return: (nn.Module) Adaptation network parameters for outputting classification parameters.
        """
        return DenseResidualBlock(in_size, out_size)

    def forward(self, representation_dict):
        """
        Forward pass through adaptation network. Returns classification parameters for task.
        :param representation_dict: (dict::torch.tensors) Dictionary containing class-level representations for each
                                    class in the task.
        :return: (dict::torch.tensors) Dictionary containing the weights and biases for the classification of each class
                 in the task. Model can extract parameters and build the classifier accordingly. Supports sampling if
                 ML-PIP objective is desired.
        """
        classifier_param_dict = {}
        class_weight_means = []
        class_bias_means = []

        # Extract and sort the label set for the task
        label_set = list(representation_dict.keys())
        label_set.sort()
        num_classes = len(label_set)

        # For each class, extract the representation and pass it through adaptation network to generate classification
        # params for that class. Store parameters in a list,
        for class_num in label_set:
            nu = representation_dict[class_num]
            class_weight_means.append(self.weight_means_processor(nu))
            class_bias_means.append(self.bias_means_processor(nu))

        # Save the parameters as torch tensors (matrix and vector) and add to dictionary
        classifier_param_dict['weight_mean'] = torch.cat(class_weight_means, dim=0)
        classifier_param_dict['bias_mean'] = torch.reshape(torch.cat(class_bias_means, dim=1), [num_classes, ])

        return classifier_param_dict


class CrossTransformerClassifierAdaptationNetwork(nn.Module):
    """
    Versa-style adaptation network for linear classifier (see https://arxiv.org/abs/1805.09921 for full details).
    :param d_theta: (int) Input / output feature dimensionality for layer.
    """
    def __init__(self, d_theta):
        super(CrossTransformerClassifierAdaptationNetwork, self).__init__()
        self.weight_means_processor = self._make_mean_dense_block(d_theta, d_theta)
        self.bias_means_processor = self._make_mean_dense_block(d_theta, 1)

    @staticmethod
    def _make_mean_dense_block(in_size, out_size):
        """
        Simple method for generating different types of blocks. Final code only uses dense residual blocks.
        :param in_size: (int) Input representation dimensionality.
        :param out_size: (int) Output representation dimensionality.
        :return: (nn.Module) Adaptation network parameters for outputting classification parameters.
        """
        return DenseResidualBlock(in_size, out_size)

    def forward(self, prototypes):

        b, k, e = prototypes.shape  # b: batch size, k: num classes, e: embedding size

        prototypes = rearrange(prototypes, 'b k e -> (b k) e')
        class_weight_means = self.weight_means_processor(prototypes)
        class_bias_means = self.bias_means_processor(prototypes)

        return {
            'weight_mean': rearrange(class_weight_means, '(b k) e -> b k e', b = b, k = k),
            'bias_mean': rearrange(class_bias_means, '(b k) e -> b k e', b = b, k = k)
        }


class AttentiveLinearClassifierAdaptationNetwork(LinearClassifierAdaptationNetwork):
    def forward(self, representation_dict):
        classifier_param_dict = {}
        class_weight_means = []
        class_bias_means = []

        # Extract and sort the label set for the task
        label_set = list(representation_dict.keys())
        label_set.sort()
        num_classes = len(label_set)

        # For each class, extract the representation and pass it through adaptation network to generate classification
        # params for that class. Store parameters in a list,
        for class_num in label_set:
            nu = representation_dict[class_num]
            class_weight_means.append(self.weight_means_processor(nu))
            class_bias_means.append(self.bias_means_processor(nu))

        # Save the parameters as torch tensors (matrix and vector) and add to dictionary
        classifier_param_dict['weight_mean'] = torch.stack(class_weight_means, dim=1)
        classifier_param_dict['bias_mean'] = torch.stack(class_bias_means, dim=1)

        return classifier_param_dict
