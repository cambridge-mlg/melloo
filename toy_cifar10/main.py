import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import os
import numpy as np
import pickle
import protonets


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="1"


root = '/scratch/etv21/cifar10_data'


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

num_epochs = 1
batch_size = 40
learning_rate = 0.001

flip_fraction = 0.4

horse_index = 7
automobile_index = 1
keep_class_indices = [horse_index, automobile_index]
num_classes = len(keep_class_indices)

def cross_entropy(logits, labels):
    return torch.nn.functional.cross_entropy(logits, labels)

def get_indices_for_classes(target_tensor, class_indices):
    binary_mask = torch.zeros(target_tensor.shape)
    normalized_targets = torch.ones(target_tensor.shape)*(-1)
    
    for ci, class_index in enumerate(class_indices):
        mask = torch.where(target_tensor == class_index, torch.ones(1), torch.zeros(1)).bool()
        binary_mask = binary_mask + mask
        normalized_targets[mask] = ci
        
    binary_mask = binary_mask.numpy()
    keep_indices = [x for x in range(0, len(binary_mask)) if binary_mask[x] == 1]
    return keep_indices, normalized_targets.int().tolist()
        

def make_data_loader(batch_size, train=True, shuffle=False):

    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize( 
           (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) 
        )
    ])

    dataset = torchvision.datasets.CIFAR10(
    root = root, train=train,
    download =True, transform=transform)
    
    target_tensor = torch.tensor(dataset.targets)
    keep_indices, normalized_targets = get_indices_for_classes(target_tensor, keep_class_indices)
    # Now we can use the subset dataset
    dataset.targets = normalized_targets
    binary_dataset = torch.utils.data.Subset(dataset, keep_indices)
    
    return torch.utils.data.DataLoader(binary_dataset, batch_size=batch_size, shuffle=shuffle)

# Warning: mutates the model
def get_feature_embeddings(model, data_loader):
        
    # Remove last fully-connected layer so we access the feature embeddings instead
    new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    model.classifier = new_classifier
    model = model.to(device)
    
    img_embeddings = None
    train_labels = None
    
    with torch.no_grad():
        for i, (imgs , lbls) in enumerate(data_loader):
            images = imgs.to(device)
            labels = lbls.to(device)
        
            # Grab features from model
            batch_embeddings = model(images)
            
            # Preserve 
            if img_embeddings is None:
                img_embeddings = batch_embeddings.to('cpu').numpy()
                train_labels = lbls.numpy()
            else:
                img_embeddings = np.append(img_embeddings, batch_embeddings.to('cpu').numpy(), axis=0)
                train_labels = np.append(train_labels, lbls.numpy())

    return img_embeddings, train_labels
    
    
def train_model_head(model, data_loader):
    input_last_layer = model.classifier[6].in_features
    # Newly created final layer will automatically have gradients enabled
    model.classifier[6] = nn.Linear(input_last_layer, num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9,weight_decay=5e-4)

    
    # Train
    for epoch in range(num_epochs):
        for i, (imgs , labels) in enumerate(data_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels_hat = model(imgs)
            n_corrects = (labels_hat.argmax(axis=1)==labels).sum().item()
            loss_value = criterion(labels_hat, labels)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (i+1) % 250 == 0:
                print(f'epoch {epoch+1}/{num_epochs}, step: {i+1}/{n_total_step}: loss = {loss_value:.5f}, acc = {100*(n_corrects/labels.size(0)):.2f}%')
                
def test_model(model, data_loader):                
    # What we want to do now is generate feature embeddings for all of our inputs:
    with torch.no_grad():
        number_corrects = 0
        number_samples = 0
        for i, (imgs , labels) in enumerate(data_loader):
            images_set = imgs.to(device)
            labels_set = labels.to(device)
        
            y_predicted = model(images_set)
            labels_predicted = y_predicted.argmax(axis = 1)
            number_corrects += (labels_predicted==labels_set).sum().item()
            number_samples += labels_set.size(0)
            
        print(f'Overall accuracy {(number_corrects / number_samples)*100}%')

        # Save out embeddings

def save_embeddings(features, labels, path):
    pickle.dump(features, open(os.path.join(path, "embeddings.pickle"), "wb"))
    pickle.dump(labels, open(os.path.join(path, "labels.pickle"), "wb"))
    
def load_embeddings(path):
    features = pickle.load(open(os.path.join(path, "embeddings.pickle"), "rb"))
    labels = pickle.load(open(os.path.join(path, "labels.pickle"), "rb"))    
    return features, labels
   

def initialize_embeddings(root, train):
    if train:
        key = "train"
    else:
        key = "test"
    output_dir = os.path.join(root, key)
    if os.path.exists(output_dir):
        return load_embeddings(output_dir)
    else:
        os.makedirs(output_dir)
        data_loader = make_data_loader(batch_size, train=train, shuffle=False)
        # We freeze all the VGG layers except the top layer
        model = models.vgg16(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        features, labels = get_feature_embeddings(model, data_loader)
        save_embeddings(features, labels, output_dir)
        return features, labels

def calculate_rankings(protonet, support_features, support_labels, query_features, way):

    # Calculate loo weights
    weights = torch.zeros(len(support_features))

    
    if len(support_features) != len(support_labels):
        import pdb; pdb.set_trace()
    for i in range(0, len(support_features)):
        loo_features = support_features[i].unsqueeze(0)
        loo_labels = support_labels[i].unsqueeze(0)
        logits_loo = protonet.loo(loo_features, loo_labels, query_features, way)
        loss = cross_entropy(logits_loo, test_labels)
        weights[i] = loss

    rankings = torch.argsort(weights, descending=True)
    return rankings

train_features, train_labels = initialize_embeddings(root, train=True)
test_features, test_labels = initialize_embeddings(root, train=False)

num_train = 10
train_features = train_features[0:num_train]
train_labels = train_labels[0:num_train]

train_features, test_features = torch.from_numpy(train_features).to(device), torch.from_numpy(test_features).to(device)
train_labels = torch.from_numpy(train_labels).type(torch.LongTensor).to(device)
test_labels = torch.from_numpy(test_labels).type(torch.LongTensor).to(device)



# Clean performance

protonet = protonets.ProtoNets(num_classes)

logits = protonet(train_features, train_labels, test_features)
predictions = logits.argmax(axis=1)
acc = (predictions==test_labels).sum().item()/float(len(predictions))
print(f'Overall accuracy {(acc)*100}%')
loss = cross_entropy(logits, test_labels)
print(f'Overall loss {(loss)}')

import pdb; pdb.set_trace()

# Flip label experiment

flipped_train_labels = train_labels.clone()
indices_to_flip = torch.randperm(len(train_labels))[0:int(len(train_labels)*flip_fraction)]
flipped_train_labels[indices_to_flip] = 1 - flipped_train_labels[indices_to_flip]

logits = protonet(train_features, flipped_train_labels, test_features)
predictions = logits.argmax(axis=1)
acc = (predictions==test_labels).sum().item()/float(len(predictions))
print(f'40% Noisy accuracy {(acc)*100}%')
loss = cross_entropy(logits, test_labels)
print(f'40% Noisy loss {(loss)}')

rankings = calculate_rankings(protonet, train_features, flipped_train_labels, test_features, num_classes)
num_to_keep = int(len(train_features) * (1 - flip_fraction ))
keep_indices = rankings[0:num_to_keep]
relabel_indices = rankings[num_to_keep:]

import pdb; pdb.set_trace()

num_correct_indices = len(set(indices_to_flip.numpy()).intersection(set(relabel_indices.numpy())))
print(f'Correctly identified indices {(num_correct_indices)} out of {(flip_fraction*len(train_features))}')

relabeled_train_labels = flipped_train_labels.clone()
relabeled_train_labels[relabel_indices] = train_labels[relabel_indices]

logits = protonet(train_features, relabeled_train_labels, test_features)
predictions = logits.argmax(axis=1)
acc = (predictions==test_labels).sum().item()/float(len(predictions))
print(f'Relabeled accuracy {(acc)*100}%')
loss = cross_entropy(logits, test_labels)
print(f'Relabeled loss {(loss)}')
