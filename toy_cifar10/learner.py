import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import os
import numpy as np


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


horse_index = 7
automobile_index = 1
keep_class_indices = [horse_index, automobile_index]
num_classes = len(keep_class_indices)

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
    import pdb; pdb.set_trace()
    
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

    import pdb; pdb.set_trace()
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


#TODO: I'm not convinced we can use the default data loader, as we need to keep track of ids. 

train_loader = make_data_loader(batch_size, train=True, shuffle=False)
test_loader = make_data_loader(batch_size, train=False, shuffle=True)

n_total_step = len(train_loader)
print(n_total_step)


# We freeze all the VGG layers except the top layer
model = models.vgg16(pretrained = True)
for param in model.parameters():
    param.requires_grad = False
    
    
feature_embeddings, labels = get_feature_embeddings(model, train_loader)
pickle.dump(feature_embeddings, open(root + "/embeddings.pickle", "wb"))
pickle.dump(labels, open(root + "/labels.pickle", "wb"))
