import torch

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