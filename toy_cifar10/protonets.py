import torch
import torch.nn as nn
import numpy as np
from helper_classes import euclidean_metric

cluster_distance = 800
    
class ProtoNets(nn.Module):
    """
    Main model class.
    :param args: (Argparser) Arparse object containing model hyper-parameters.
    """
    def __init__(self, way, scale_by_std=False):
        super(ProtoNets, self).__init__()
        self.way = way
        self.scale_by_std = scale_by_std

    def forward(self, context_features, context_labels, target_features):
        """
        Forward pass through the model for one episode.
        :param context_images: (torch.tensor) Images in the context set (batch x C x H x W).
        :param context_labels: (torch.tensor) Labels for the context set (batch x 1 -- integer representation).
        :param target_images: (torch.tensor) Images in the target set (batch x C x H x W).
        :return: (torch.tensor) Categorical distribution on label set for each image in target set (batch x num_labels).
        """
        way = self.way
        prototypes, stds = self._compute_prototypes(context_features, context_labels, way)

        print("Prototype distances: {}".format(euclidean_metric(prototypes, prototypes)[0][1].item()))
        return self.predict(target_features)
        
    # Will always used saved protoypes and stds
    def predict(self, target_features):
        return self._predict(target_features, self.prototypes, self.stds)
        
    # Will use specified prototypes and stds
    def _predict(self, target_features, class_prototypes, class_stds):
        logits = euclidean_metric(target_features, class_prototypes)
        # Scale by squared standard deviation
        if self.scale_by_std:
            logits = logits/(class_stds**2).to(logits.device)
        return logits

    def _compute_prototypes(self, context_features, context_labels, way, save_prototypes=True):
        prototypes = []
        prototype_labels = []
        prototype_counts = []
        cluster_stds = []
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

            class_std = torch.std(class_features)
            #class_std = torch.Tensor([1]).to(context_features.device) #torch.std(class_features)
            prototypes.append(class_mean)
            cluster_stds.append(class_std)
            prototype_labels.append(c)
            prototype_counts.append(len(class_features))

        prototypes = torch.squeeze(torch.stack(prototypes))
        cluster_stds = torch.Tensor(cluster_stds)
        if save_prototypes:
            self.prototypes = prototypes
            self.prototype_labels = prototype_labels
            self.prototype_counts = prototype_counts
            self.stds = cluster_stds
        
        return prototypes, cluster_stds
        
        
    def loo(self, loo_features, loo_labels, target_features, way):
        if self.scale_by_std:
            return self._loo_std_scale(loo_features, loo_labels, target_features, way)
        else:
            return self._loo_efficient(loo_features, loo_labels, target_features, way)
        
    # Subtract the given points from the means. i.e. given points should be left out
    # Doesn't account for potential change in stds
    def _loo_efficient(self, loo_features, loo_labels, target_features, way):
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

        logits = self._predict(target_features, prototypes, self.stds)
        return logits
        
        
    # Recalculate all values with the given points. i.e. points are ones remaining after the ones to be left out have been removed
    def _loo_std_scale(self, loo_features, loo_labels, target_features, way):
        new_prototypes, new_stds = self._compute_prototypes(loo_features, loo_labels, way, save_prototypes=False)
        logits = self._predict(target_features, new_prototypes, new_stds)
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
