import torch
from torch.utils.data import Dataset

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs
        
class EmbeddedDataset(Dataset):
    def __init__(self, feature_embeddings, labels):
        self.feature_embeddings = feature_embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.feature_embeddings[idx], self.labels[idx]

def euclidean_metric(target_features, class_prototypes):
    num_target_features = target_features.shape[0]
    num_prototypes = class_prototypes.shape[0]

    distances = (target_features.unsqueeze(1).expand(num_target_features, num_prototypes, -1) -
                 class_prototypes.unsqueeze(0).expand(num_target_features, num_prototypes, -1)).pow(2).sum(dim=2)
    return -distances
    