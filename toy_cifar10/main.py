import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import os
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy.random as rand

from helper_classes import LogisticRegression, EmbeddedDataset
from plot_decision_regions import PlotSettings, plot_decision_regions
import protonets


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="1"

root = '/scratch/etv21/cifar10_data'


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

num_epochs = 5
batch_size = 40
learning_rate = 0.001

flip_fraction = 0.4
check_fractions = [flip_fraction] #[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
toy_is_noisy = False

logfile = open(os.path.join(root, "log.txt"), 'a+')


horse_index = 7
automobile_index = 1
keep_class_indices = [horse_index, automobile_index]
num_classes = len(keep_class_indices)

logfile.write("=======================\nToy Protonets\n====================\n")
logfile.write(f'Params: flip_fraction: {flip_fraction}, toy is noisy: {toy_is_noisy}, num_classes: {num_classes}, num_epochs: {num_epochs}, batch_size: {batch_size}, learning_rate: {learning_rate}\n')


def cross_entropy(logits, labels, fig_prefix=None):
    unreduced_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    if False: # fig_prefix is not None:
        plt.hist(unreduced_loss.cpu().numpy(), 50, density=True, range=(0, 0.5))
        plt.xlabel('Loss')
        plt.title('Histogram of test losses (before mean)')
        plt.grid(True)
        plt.savefig(os.path.join(root, fig_prefix + "_loss_hist.pdf"))
        plt.close()
    return unreduced_loss.mean()

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
    
    
def train_model(model, data_loader, reset_head=False, hush=False):
    if reset_head:
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
            if not hush and epoch == num_epochs-1 and (i+1) % 250 == 0:
                print(f'epoch {epoch+1}/{num_epochs}: loss = {loss_value:.5f}, acc = {100*(n_corrects/labels.size(0)):.2f}%')
                logfile.write(f'epoch {epoch+1}/{num_epochs}: loss = {loss_value:.5f}, acc = {100*(n_corrects/labels.size(0)):.2f}%\n')
    return model
                
def test_model(model, data_loader, hush=False):                
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
            
        if not hush:
            print(f'Logistic regression accuracy {(number_corrects / number_samples)*100}%')
            logfile.write(f'Logistic regression accuracy {(number_corrects / number_samples)*100}%\n')
        return (number_corrects / number_samples)*100


def save_embeddings(features, labels, path):
    pickle.dump(features, open(os.path.join(path, "embeddings.pickle"), "wb"))
    pickle.dump(labels, open(os.path.join(path, "labels.pickle"), "wb"))
    
def load_embeddings(path):
    features = pickle.load(open(os.path.join(path, "embeddings.pickle"), "rb"))
    labels = pickle.load(open(os.path.join(path, "labels.pickle"), "rb"))    
    return features, labels
   

def initialize_embeddings(root, train, toy=False):
    if train:
        key = "train"
    else:
        key = "test"
    output_dir = os.path.join(root, key)
    
    if toy:
        return generate_gaussian_data(train)

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

def generate_gaussian_data(train=True, train_error_rate= 0.01, test_error_rate = 0.024):
    train_per_class = 5000
    test_per_class = 1000
    
    if train:
        num_instances = train_per_class
        error_rate = train_error_rate
    else:
        num_instances = test_per_class
        error_rate = test_error_rate
    if not toy_is_noisy:
        error_rate = 0

    unpertured_prototype_distance = np.sqrt(384.6158)/2.0
    unperturbed_symmetric_coords = unpertured_prototype_distance/np.sqrt(2)
    prototype_0 = np.array([unperturbed_symmetric_coords, unperturbed_symmetric_coords])
    prototype_1 = np.array([-unperturbed_symmetric_coords, -unperturbed_symmetric_coords])
    '''
    if flip_fraction == 0.4:
        perturbed_prototype_distance = 16.1239
    else:
        perturbed_prototype_distance = 1.0131
    perturbed_symmetric_coords = perturbed_prototype_distance/np.sqrt(2)
    pert_prototype_0 = np.array(perturbed_symmetric_coords, perturbed_symmetric_coords)
    pert_prototype_1 = np.array(-perturbed_symmetric_coords, -perturbed_symmetric_coords)
    '''
    
    points = rand.normal(loc=prototype_0, size=(num_instances, 2)).astype('f')
    points = np.append(points, rand.normal(loc=prototype_1, size=(num_instances, 2)).astype('f'), axis=0)
    labels_0 = np.append(np.zeros(int(num_instances*(1 - error_rate))), np.ones(int(num_instances*error_rate)))
    labels_1 = np.append(np.ones(int(num_instances*(1 - error_rate))), np.zeros(int(num_instances*error_rate)))
    return points, np.append(labels_0, labels_1, axis=0)


def calculate_rankings(protonet, support_features, support_labels, query_features, query_labels, way):
    # Calculate loo weights
    weights = torch.zeros(len(support_features))

    
    if len(support_features) != len(support_labels):
        import pdb; pdb.set_trace()
    for i in range(0, len(support_features)):
        loo_features = support_features[i].unsqueeze(0)
        loo_labels = support_labels[i].unsqueeze(0)
        logits_loo = protonet.loo(loo_features, loo_labels, query_features, way)
        loss = cross_entropy(logits_loo, query_labels)
        weights[i] = loss

    rankings = torch.argsort(weights, descending=True)
    return rankings

# Protonets: {flip_fraction*100}% Noisy
def print_accuracy(logits, test_labels, descrip, hush=False):
    predictions = logits.argmax(axis=1)
    acc = (predictions==test_labels).sum().item()/float(len(predictions))
    loss = cross_entropy(logits, test_labels)

    if not hush:
        print(f'{descrip} accuracy {(acc)*100}%')
        print(f'{descrip} loss {(loss)}')
    logfile.write(f'{descrip} accuracy {(acc)*100}%\n')
    logfile.write(f'{descrip} loss {(loss)}\n')
    return acc, loss

def train_logistic_regression_head(train_features, train_labels, test_features, test_labels):

    clean_embedding_dataset_test = EmbeddedDataset(test_features, test_labels)
    clean_dataloader_test = torch.utils.data.DataLoader(clean_embedding_dataset_test, batch_size=batch_size, shuffle=False)

    clean_embedding_dataset_train = EmbeddedDataset(train_features, train_labels)
    clean_dataloader_train = torch.utils.data.DataLoader(clean_embedding_dataset_train, batch_size=batch_size, shuffle=False)

    clean_lreg = LogisticRegression(train_features.shape[1], num_classes)
    clean_lreg = train_model(clean_lreg, clean_dataloader_train)
    return test_model(clean_lreg, clean_dataloader_test)

train_features, train_labels = initialize_embeddings(root, train=True, toy=True)
test_features, test_labels = initialize_embeddings(root, train=False, toy=True)

#num_train = 10
#train_features = train_features[0:num_train]
#train_labels = train_labels[0:num_train]

train_features, test_features = torch.from_numpy(train_features).to(device), torch.from_numpy(test_features).to(device)
train_labels = torch.from_numpy(train_labels).type(torch.LongTensor).to(device)
test_labels = torch.from_numpy(test_labels).type(torch.LongTensor).to(device)

subset_indices = torch.LongTensor([0, 1, 2, 3, 4, 998, 999, 1000, 1001, 1002, 1003, 1004, 1998, 1999])
test_labels_subset = test_labels[subset_indices]
test_features_subset = test_features[subset_indices]

confusing_test_features = torch.concat((test_features[1000 - int(0.024*1000):1000], test_features[2000 - int(0.024*1000):]), dim=0)
confusing_labels = torch.concat((test_labels[1000 - int(0.024*1000):1000], test_labels[2000 - int(0.024*1000):]), dim=0)
easy_test_features = torch.concat((test_features[0:1000 - int(0.024*1000)], test_features[1000:2000 - int(0.024*1000)]), dim=0)
easy_labels = torch.concat((test_labels[0:1000 - int(0.024*1000)], test_labels[1000:2000 - int(0.024*1000)]), dim=0)

# Clean performance on protonets
plot_config = PlotSettings()
protonet = protonets.ProtoNets(num_classes)

logits = protonet(train_features, train_labels, test_features)
print_accuracy(logits, test_labels, "Protonets: clean")
plot_decision_regions(protonet.prototypes, test_features_subset, test_labels_subset, "clean.pdf", plot_config, protonet, device)

confusing_logits = protonet.classify(confusing_test_features)
print_accuracy(confusing_logits, confusing_labels, "Protonets, confusing")
easy_logits = protonet.classify(easy_test_features)
print_accuracy(easy_logits, easy_labels, "Protonets, confusing")

#train_logistic_regression_head(train_features, train_labels, test_features, test_labels)

# Flip label experiment

flipped_train_labels = train_labels.clone()
indices_to_flip = torch.randperm(len(train_labels))[0:int(len(train_labels)*flip_fraction)]
flipped_train_labels[indices_to_flip] = 1 - flipped_train_labels[indices_to_flip]
#print("Flipped sum: {}".format(flipped_train_labels.sum()))

#flipped_test_labels = test_labels.clone()
#test_indices_to_flip = torch.randperm(len(test_labels))[0:int(len(test_labels)*flip_fraction)]
#flipped_test_labels[test_indices_to_flip] = 1 - flipped_test_labels[test_indices_to_flip]

logits = protonet(train_features, flipped_train_labels, test_features)
print_accuracy(logits, test_labels, f'Protonets: {flip_fraction*100}% Noisy')
plot_decision_regions(protonet.prototypes, test_features_subset, test_labels_subset, "noisy_{}.pdf".format(flip_fraction*100), plot_config, protonet, device)

#train_logistic_regression_head(train_features, flipped_train_labels, test_features, test_labels)

rankings = calculate_rankings(protonet, train_features, flipped_train_labels, test_features, test_labels, num_classes)
#rankings = calculate_rankings(protonet, train_features, flipped_train_labels, test_features, flipped_test_labels, num_classes)

num_correctly_identified = []
relabelled_protonets_acc = []
relabelled_protonets_loss = []
relabelled_lreg_acc = []

for check_fraction in check_fractions:
    num_to_keep = int(len(train_features) * (1 - check_fraction ))
    keep_indices = rankings[0:num_to_keep]
    relabel_indices = rankings[num_to_keep:]
    num_correct_indices = len(set(indices_to_flip.numpy()).intersection(set(relabel_indices.numpy())))
    num_correctly_identified.append(num_correct_indices)
    #print(f'Correctly identified indices {(num_correct_indices)} out of {(flip_fraction*len(train_features))}')

    relabeled_train_labels = flipped_train_labels.clone()
    relabeled_train_labels[relabel_indices] = train_labels[relabel_indices]

    logits = protonet(train_features, relabeled_train_labels, test_features)
    acc, loss = print_accuracy(logits, test_labels, 'Protonets: {check_fraction*100}% relabeled', hush=True)
    relabelled_protonets_acc.append(acc)
    relabelled_protonets_loss.append(loss.item())

    #test_acc = train_logistic_regression_head(train_features, relabeled_train_labels, test_features, test_labels)
    #relabelled_lreg_acc.append(test_acc)

logfile.write("Check fractions: {}\n".format(check_fractions))
logfile.write("Num correctly identified: {}\n".format(num_correctly_identified))
logfile.write("Relabelled protonets acc: {}\n".format(relabelled_protonets_acc))
logfile.write("Relabelled protonets loss: {}\n".format(relabelled_protonets_loss))
logfile.write("Relabelled LReg acc: {}\n".format(relabelled_lreg_acc))

