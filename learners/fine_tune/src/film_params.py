import torch
import torch.nn as nn


class FilmAdapter(nn.Module):
    def __init__(self, layer, num_maps, num_blocks):
        super().__init__()
        self.num_maps = num_maps
        self.num_blocks = num_blocks
        self.num_target_layers = len(self.num_maps)
        self.layer = layer
        self.layers = self.get_layers()

    def get_layers(self):
        layers = nn.ModuleList()
        for num_maps, num_blocks in zip(self.num_maps, self.num_blocks):
            layers.append(
                self.layer(
                    num_maps=num_maps,
                    num_blocks=num_blocks,
                )
            )
        return layers

    def forward(self, x):
        return [self.layers[layer](x) for layer in range(self.num_target_layers)]

    def regularization_term(self):
        l2_term = 0
        for layer in self.layers:
            l2_term += layer.regularization_term()
        return l2_term


class FilmLayer(nn.Module):
    def __init__(self, num_maps, num_blocks):
        super(FilmLayer, self).__init__()

        self.num_maps = num_maps
        self.num_blocks = num_blocks

        self.gamma = nn.ParameterList()
        self.beta = nn.ParameterList()
        self.gamma_regularizers = nn.ParameterList()
        self.beta_regularizers = nn.ParameterList()

        for i in range(self.num_blocks):
            self.gamma.append(nn.Parameter(torch.ones(self.num_maps[i]), requires_grad=True))
            self.beta.append(nn.Parameter(torch.zeros(self.num_maps[i]), requires_grad=True))
            self.gamma_regularizers.append(nn.Parameter(nn.init.normal_(torch.empty(num_maps[i]), 0, 0.001),
                                                        requires_grad=True))
            self.beta_regularizers.append(nn.Parameter(nn.init.normal_(torch.empty(num_maps[i]), 0, 0.001),
                                                       requires_grad=True))

    def forward(self, x):
        block_params = []
        for block in range(self.num_blocks):
            block_param_dict = {
                'gamma': self.gamma[block] * self.gamma_regularizers[block] +
                         torch.ones_like(self.gamma_regularizers[block]),
                'beta': self.beta[block] * self.beta_regularizers[block]
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
        for gamma_regularizer, beta_regularizer in zip(self.gamma_regularizers, self.beta_regularizers):
            l2_term += (gamma_regularizer ** 2).sum()
            l2_term += (beta_regularizer ** 2).sum()
        return l2_term


class FilmLayerResnet(nn.Module):
    """
    Single adaptation network for generating the parameters of each layer in the base network. Will be wrapped around
    by FilmAdaptationNetwork.
    :param num_maps: (int) Number of output maps to be adapted in base network layer.
    :param num_blocks: (int) Number of blocks being adapted in the base network layer.
    :param z_g_dim: (int) Dimensionality of input to network (task level representation).
    """
    def __init__(self, num_maps, num_blocks):
        super().__init__()
        self.num_maps = num_maps
        self.num_blocks = num_blocks

        # Initialize the processors (adaptation networks) and regularization lists for each of the output params
        self.gamma1, self.gamma1_regularizers = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.gamma2, self.gamma2_regularizers = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.beta1, self.beta1_regularizers = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.beta2, self.beta2_regularizers = torch.nn.ParameterList(), torch.nn.ParameterList()

        # Generate the required layers / regularization parameters, and collect them in ModuleLists and ParameterLists
        for _ in range(self.num_blocks):
            self.gamma1.append(nn.Parameter(torch.ones(self.num_maps), requires_grad=True))
            self.gamma1_regularizers.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(num_maps), 0, 0.001),
                                                               requires_grad=True))

            self.beta1.append(nn.Parameter(torch.zeros(self.num_maps), requires_grad=True))
            self.beta1_regularizers.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(num_maps), 0, 0.001),
                                                              requires_grad=True))

            self.gamma2.append(nn.Parameter(torch.ones(self.num_maps), requires_grad=True))
            self.gamma2_regularizers.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(num_maps), 0, 0.001),
                                                               requires_grad=True))

            self.beta2.append(nn.Parameter(torch.zeros(self.num_maps), requires_grad=True))
            self.beta2_regularizers.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(num_maps), 0, 0.001),
                                                              requires_grad=True))

    def forward(self, x):
        """
        Forward pass through adaptation network.
        :param x: (torch.tensor) Input representation to network (task level representation z).
        :return: (list::dictionaries) Dictionary for every block in layer. Dictionary contains all the parameters
                 necessary to adapt layer in base network. Base network is aware of dict structure and can pull params
                 out during forward pass.
        """
        block_params = []
        for block in range(self.num_blocks):
            block_param_dict = {
                'gamma1': self.gamma1[block] * self.gamma1_regularizers[block] +
                          torch.ones_like(self.gamma1_regularizers[block]),
                'beta1': self.beta1[block] * self.beta1_regularizers[block],
                'gamma2': self.gamma2[block] * self.gamma2_regularizers[block] +
                          torch.ones_like(self.gamma2_regularizers[block]),
                'beta2': self.beta2[block] * self.beta2_regularizers[block]
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
