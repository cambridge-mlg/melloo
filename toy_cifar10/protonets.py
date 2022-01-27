import torch
import torch.nn as nn
import numpy as np

cluster_distance = 800

def euclidean_metric(target_features, class_prototypes):
    num_target_features = target_features.shape[0]
    num_prototypes = class_prototypes.shape[0]

    distances = (target_features.unsqueeze(1).expand(num_target_features, num_prototypes, -1) -
                 class_prototypes.unsqueeze(0).expand(num_target_features, num_prototypes, -1)).pow(2).sum(dim=2)
    return -distances
    
class ProtoNets(nn.Module):
    """
    Main model class.
    :param args: (Argparser) Arparse object containing model hyper-parameters.
    """
    def __init__(self, way):
        super(ProtoNets, self).__init__()
        self.way = way

    def forward(self, context_features, context_labels, target_features):
        """
        Forward pass through the model for one episode.
        :param context_images: (torch.tensor) Images in the context set (batch x C x H x W).
        :param context_labels: (torch.tensor) Labels for the context set (batch x 1 -- integer representation).
        :param target_images: (torch.tensor) Images in the target set (batch x C x H x W).
        :return: (torch.tensor) Categorical distribution on label set for each image in target set (batch x num_labels).
        """

        way = self.way
        prototypes = self._compute_prototypes(context_features, context_labels, way)

        print("Prototype distances: {}".format(euclidean_metric(prototypes, prototypes)[0][1].item()))
        logits = euclidean_metric(target_features, self.prototypes)
        return logits
        
    def classify(self, target_features):
        logits = euclidean_metric(target_features, self.prototypes)
        # Scale by squared standard deviation
        #logits = logits/(self.std_devs**2).to(logits.device)
        return logits

    def _compute_prototypes(self, context_features, context_labels, way):
        prototypes = []
        prototype_labels = []
        prototype_counts = []
        cluster_std_devs = []
        #unpertured_prototype_distance = np.sqrt(cluster_distance)/2.0
        #unperturbed_symmetric_coords = unpertured_prototype_distance/np.sqrt(2)
        #prototype_0 = np.array([unperturbed_symmetric_coords, unperturbed_symmetric_coords])
        #prototype_1 = np.array([-unperturbed_symmetric_coords, -unperturbed_symmetric_coords])
        for c in torch.unique(context_labels):
            class_features = torch.index_select(context_features, 0, self._extract_class_indices(context_labels, c))
            class_mean = torch.mean(class_features, dim=0, keepdim=True)
            '''
            if c == 0:
                class_mean = torch.Tensor(prototype_0).unsqueeze(0).to(context_features.device)
            else:
                class_mean = torch.Tensor(prototype_1).unsqueeze(0).to(context_features.device)
            '''

            class_stddev = torch.std(class_features)
            #class_stddev = torch.Tensor([1]).to(context_features.device) #torch.std(class_features)
            prototypes.append(class_mean)
            cluster_std_devs.append(class_stddev)
            prototype_labels.append(c)
            prototype_counts.append(len(class_features))

        self.prototypes = torch.squeeze(torch.stack(prototypes))
        self.prototype_labels = prototype_labels
        self.prototype_counts = prototype_counts
        self.std_devs = torch.Tensor(cluster_std_devs)
        
        return self.prototypes
        
    def loo(self, loo_features, loo_labels, target_features, way):
        prototypes = self.prototypes.clone()
        # Remove the given features from the computed prototypes
        for c in torch.unique(loo_labels):
            # Find corresponding prototype:
            prototype = None
            prototype_count = -1
            prototype_index = -1
            other_prototype = None
            other_prototype_count = -1
            other_prototype_index = -1
            assert len(self.prototype_labels) == 2
            for index, label in enumerate(self.prototype_labels):
                if label == c:
                    prototype = prototypes[index].squeeze()
                    prototype_count = self.prototype_counts[index]
                    prototype_index = index
                else:
                    other_prototype = prototypes[index].squeeze()
                    other_prototype_count = self.prototype_counts[index]
                    other_prototype_index = index
                
                    
            if prototype is None:
                print("Failed to find prototype for label %".format(c))
                return
                
            class_features = torch.index_select(loo_features, 0, self._extract_class_indices(loo_labels, c))
            prototype = ((prototype * prototype_count) - torch.sum(class_features, dim=0)) / (prototype_count - len(loo_features))
            prototypes[prototype_index] = prototype
            other_prototype = ((other_prototype * other_prototype_count) + torch.sum(class_features, dim=0)) / (other_prototype_count + len(loo_features))
            prototypes[other_prototype_index] = other_prototype

        logits = euclidean_metric(target_features, prototypes)
        return logits

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
