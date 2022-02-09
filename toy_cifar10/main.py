import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg
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

from mahalanobis import MahalanobisPredictor
from helper_classes import LogisticRegression, EmbeddedDataset
from plot_decision_regions import PlotSettings, plot_decision_regions
import protonets
from helper_classes import euclidean_metric
from tqdm import tqdm
from argparse import ArgumentParser


#torch.use_deterministic_algorithms(True)
torch.manual_seed(2) #0
rand.seed(20160704) #20160702

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


parser = ArgumentParser()
parser.add_argument("-r", "--root", default="/scratch/etv21/debug", help="root where output will be written to")
parser.add_argument("-dr", "--data_root", default='/scratch/etv21/cifar10_data', help="Directory where data/embeddings will be looked for")




parser.add_argument("--flip_fraction", type=float, default=0.2, help="Fraction of context set labels to flip")
parser.add_argument("--scale_logits", action='store_true', help="Whether to scale protonet logits by std dev")
parser.add_argument("--classifier_type", default="Protonets", choices=['Protonets', 'Mahalanobis'], help="What type of classifier to use")
parser.add_argument("--ranking_method", default="loo", choices=['loo', 'random'], help="What ranking algorithm to use")
parser.add_argument("--drop_strategy", default="Worst", choices=['None', 'Worst', "Wrong"], help="Whether to discard points from the target set and, if so, how")


parser.add_argument("--context_size", type=int, default=-1, help="Size of context set")
parser.add_argument("--target_size", type=int, default=-1, help="Size of target set")

parser.add_argument("--is_toy", action='store_true', help="Whether to use the toy gaussian dataset")
parser.add_argument("--noisy_context", action='store_true', help="Whether to put noisy labels in toy gaussian dataset's context set")
parser.add_argument("--noisy_target", action='store_true', help="Whether to put noisy labels the toy gaussian dataset's target set")
parser.add_argument("--cluster_dist", type=float, default=384.6158, help="Distance between gaussian clusters in 2-D toy case")


parser.set_defaults(scale_logits=True, noisy_context=True, noisy_target=True)

args = parser.parse_args()


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Only relevant when doing regression head
num_epochs = 5
batch_size = 40
learning_rate = 0.001

check_fractions = [args.flip_fraction] #[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

logfile = open(os.path.join(args.root, "log.txt"), 'a+')

horse_index = 7
automobile_index = 1
frog_index = 6

keep_class_indices = [horse_index, automobile_index]
num_classes = len(keep_class_indices)

logfile.write(f"=======================\nToy {args.classifier_type}\n====================\n")
logfile.write(f'Params: classifier: {args.classifier_type}, flip_fraction: {args.flip_fraction}, scale logits: {args.scale_logits}, ranking_method: {args.ranking_method}, context is noisy: {args.noisy_context}, target is noisy: {args.noisy_target}, num_classes: {num_classes}, num_epochs: {num_epochs}, batch_size: {batch_size}, learning_rate: {learning_rate}\n')


def cross_entropy(logits, labels, fig_prefix=None, return_worst=-1):
    unreduced_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")

    if fig_prefix is not None:
        plt.hist(unreduced_loss.cpu().numpy(), 100, density=True)
        plt.xlabel('Loss')
        plt.title('Histogram of test losses (max {:.3f})'.format(unreduced_loss.max()))
        plt.grid(True)
        plt.savefig(os.path.join(args.root, fig_prefix.replace(":", "-") + "_loss_hist.png"))
        #print("Top 10 loss indices: {}", worst_indices)
        plt.close()
        worst_50 =  torch.topk(unreduced_loss, 50).indices.tolist()
        plt.hist(unreduced_loss[worst_50].cpu().numpy(), 10, density=True)
        plt.xlabel('Loss')
        plt.title('Histogram of worst test losses (max {:.3f})'.format(unreduced_loss[worst_50].max()))
        plt.grid(True)
        plt.savefig(os.path.join(args.root, fig_prefix.replace(":", "-") + "_worst_loss_hist.png"))
        plt.close()
    if return_worst > 0:
        worst_indices =  torch.topk(unreduced_loss, return_worst).indices.tolist()
        return unreduced_loss.mean(), worst_indices
    else:
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
    root = data_root, train=train,
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
                print(f'epoch {epoch+1}/{num_epochs}: loss = {loss_value:.5f}, acc = {100*(n_corrects/labels.size(0)):.2f}%\n')
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
            print(f'Logistic regression accuracy {(number_corrects / number_samples)*100}%\n')
            logfile.write(f'Logistic regression accuracy {(number_corrects / number_samples)*100}%\n')
        return (number_corrects / number_samples)*100


def save_embeddings(features, labels, path):
    pickle.dump(features, open(os.path.join(path, "embeddings.pickle"), "wb"))
    pickle.dump(labels, open(os.path.join(path, "labels.pickle"), "wb"))

def load_embeddings(path):
    features = pickle.load(open(os.path.join(path, "embeddings.pickle"), "rb"))
    labels = pickle.load(open(os.path.join(path, "labels.pickle"), "rb"))
    return features, labels


def initialize_embeddings(root, train, toy=False, task_size=-1):
    if train:
        key = "train"
    else:
        key = "test"
    output_dir = os.path.join(root, key)

    if toy:
        return generate_gaussian_data(train)

    if os.path.exists(output_dir):
        features, labels = load_embeddings(output_dir)
        way = len(np.unique(labels))
        if way != num_classes:
            print("Error, embeddings don't match the requested number of classes ({} vs {})".format(way, num_classes))
            return -1
    else:
        os.makedirs(output_dir)
        data_loader = make_data_loader(batch_size, train=train, shuffle=False)
        # We freeze all the VGG layers except the top layer
        model = models.vgg16(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        features, labels = get_feature_embeddings(model, data_loader)
        save_embeddings(features, labels, output_dir)
    if task_size == -1:
        return features, labels
    else:
        assert len(features) >= task_size and len(labels) >= task_size
        return features[:task_size], labels[:task_size]

def generate_gaussian_data(train=True, train_error_rate= 0.01, test_error_rate = 0.024):
    assert args.is_toy
    train_per_class = 5000
    test_per_class = 1000

    if train:
        num_instances = train_per_class
        if args.noisy_context:
            error_rate = train_error_rate
        else:
            error_rate = 0
    else:
        num_instances = test_per_class
        if args.noisy_target:
            error_rate = test_error_rate
        else:
            error_rate = 0

    unpertured_prototype_distance = np.sqrt(args.cluster_distance)/2.0 #384.6158
    unperturbed_symmetric_coords = unpertured_prototype_distance/np.sqrt(2)
    prototype_0 = np.array([unperturbed_symmetric_coords, unperturbed_symmetric_coords])
    prototype_1 = np.array([-unperturbed_symmetric_coords, -unperturbed_symmetric_coords])

    points = rand.normal(loc=prototype_0, size=(num_instances, 2)).astype('f')
    points = np.append(points, rand.normal(loc=prototype_1, size=(num_instances, 2)).astype('f'), axis=0)
    labels_0 = np.append(np.zeros(int(num_instances*(1 - error_rate))), np.ones(int(num_instances*error_rate)))
    labels_1 = np.append(np.ones(int(num_instances*(1 - error_rate))), np.zeros(int(num_instances*error_rate)))
    return points, np.append(labels_0, labels_1, axis=0)


def calculate_random_rankings(support_labels):
    random_rankings = torch.randperm(len(support_labels))
    return random_rankings


def calculate_rankings(model, support_features, support_labels, query_features, query_labels, way):
    # Calculate loo weights
    weights = torch.zeros(len(support_features))
    full_logits = model.predict(query_features)
    full_loss = cross_entropy(full_logits, query_labels)

    worst_targets = []

    if len(support_features) != len(support_labels):
        import pdb; pdb.set_trace()
    if args.scale_logits:
        # If we're scaling, we have to recalculate means + stds with every loo
        for i in tqdm(range(0, len(support_features))):
            if i == 0:
                loo_features = support_features[1:]
                loo_labels = support_labels[1:]
            else:
                loo_features = torch.cat((support_features[0:i], support_features[i + 1:]), 0)
                loo_labels = torch.cat((support_labels[0:i], support_labels[i + 1:]), 0)
            logits_loo = model.loo(loo_features, loo_labels, query_features, way)
            if i < 100:
                loss, biggest_offending_targets = cross_entropy(logits_loo, query_labels, fig_prefix="Loo {}".format(i), return_worst=10)
            else:
                loss, biggest_offending_targets = cross_entropy(logits_loo, query_labels, return_worst=10)
            worst_targets.append(biggest_offending_targets)
            weights[i] = loss
    else:
        # If not, we can use the "efficient drop" and only specify the one we want to drop:
        for i in tqdm(range(0, len(support_features))):
            loo_features = support_features[i].unsqueeze(0)
            loo_labels = support_labels[i].unsqueeze(0)
            logits_loo = model.loo(loo_features, loo_labels, query_features, way)
            loss = cross_entropy(logits_loo, query_labels)
            weights[i] = loss

    plt.hist(worst_targets, len(query_labels), density=True)
    plt.xlabel('Index')
    plt.title('Hist of top 5 target indices with worst losses')
    plt.grid(True)
    plt.savefig(os.path.join(args.root, "worst_target_index.png"))
    plt.close()
    rankings = torch.argsort(weights, descending=True)
    return rankings

# Model: {flip_fraction*100}% Noisy
def print_accuracy(logits, test_labels, descrip, hush=False):
    predictions = logits.argmax(axis=1)
    acc = (predictions==test_labels).sum().item()/float(len(predictions))
    loss, worst_indices = cross_entropy(logits, test_labels, fig_prefix=descrip, return_worst=10)

    if not hush:
        print(f'{descrip} accuracy {(acc)*100}%')
        print(f'{descrip} loss {(loss)}')
    logfile.write(f'{descrip} accuracy {(acc)*100}%\n')
    logfile.write(f'{descrip} loss {(loss)}\n')
    return acc, loss

def print_prototype_info(prototype_dist, prototype_stds, descrip):
    #print(f'{descrip} Euclidean distace b/w prototypes {prototype_dist}')
    #print(f'{descrip} prototype stds {(prototype_stds)}')
    logfile.write(f'{descrip} Euclidean distace b/w prototypes {prototype_dist}\n')
    logfile.write(f'{descrip} prototype stds {(prototype_stds)}\n')


def train_logistic_regression_head(train_features, train_labels, test_features, test_labels):

    clean_embedding_dataset_test = EmbeddedDataset(test_features, test_labels)
    clean_dataloader_test = torch.utils.data.DataLoader(clean_embedding_dataset_test, batch_size=batch_size, shuffle=False)

    clean_embedding_dataset_train = EmbeddedDataset(train_features, train_labels)
    clean_dataloader_train = torch.utils.data.DataLoader(clean_embedding_dataset_train, batch_size=batch_size, shuffle=False)

    clean_lreg = LogisticRegression(train_features.shape[1], num_classes)
    clean_lreg = train_model(clean_lreg, clean_dataloader_train)
    return test_model(clean_lreg, clean_dataloader_test)

train_features, train_labels = initialize_embeddings(args.data_root, train=True, toy=args.is_toy, task_size=args.context_size)
test_features, test_labels = initialize_embeddings(args.data_root, train=False, toy=args.is_toy, task_size=args.target_size)


#num_train = 10
#train_features = train_features[0:num_train]
#train_labels = train_labels[0:num_train]
train_features, test_features = torch.from_numpy(train_features).to(device), torch.from_numpy(test_features).to(device)
train_labels = torch.from_numpy(train_labels).type(torch.LongTensor).to(device)
test_labels = torch.from_numpy(test_labels).type(torch.LongTensor).to(device)

if args.is_toy:
    # Select some points to plot
    #subset_indices = torch.LongTensor([0, 1, 2, 3, 4, 998, 999, 1000, 1001, 1002, 1003, 1004, 1998, 1999])
    test_labels_subset = test_labels #[subset_indices]
    test_features_subset = test_features #[subset_indices]

    if args.noisy_target:
        # The last 0.024 of each class will be "confusing" points if noisy, extract those
        confusing_test_features = torch.cat((test_features[1000 - int(0.024*1000):1000], test_features[2000 - int(0.024*1000):]), dim=0)
        confusing_labels = torch.cat((test_labels[1000 - int(0.024*1000):1000], test_labels[2000 - int(0.024*1000):]), dim=0)
        easy_test_features = torch.cat((test_features[0:1000 - int(0.024*1000)], test_features[1000:2000 - int(0.024*1000)]), dim=0)
        easy_labels = torch.cat((test_labels[0:1000 - int(0.024*1000)], test_labels[1000:2000 - int(0.024*1000)]), dim=0)
    plot_config = PlotSettings()

# Clean performance on model
if args.classifier_type == 'Protonets':
    model = protonets.ProtoNets(num_classes, scale_by_std=args.scale_logits)
else:
    model = MahalanobisPredictor()

logits = model(train_features, train_labels, test_features)
print_accuracy(logits, test_labels, f'{args.classifier_type}: initial')

if args.drop_strategy != "None":

    if args.drop_strategy == "Wrong":
        keep_mask = (logits.argmax(axis=1) == test_labels)
    elif args.drop_strategy == "Worst":
        _, worst_indices = cross_entropy(logits, test_labels, return_worst=10)
        keep_mask = torch.ones(test_labels.shape, dtype=np.bool)
        keep_mask[worst_indices] = False

    print("Dropping points that we get wrong from test set: ({} many)".format((~keep_mask).sum()))

    test_features = test_features[keep_mask]
    test_labels = test_labels[keep_mask]
    logits = model(train_features, train_labels, test_features)
    print_accuracy(logits, test_labels, f'{args.classifier_type}: all correct')


if args.classifier_type == 'Protonets':
    print_prototype_info(euclidean_metric(model.prototypes, model.prototypes), model.stds, f'{args.classifier_type}: initial')
else:
    print_prototype_info(euclidean_metric(model.prototypes, model.prototypes), torch.linalg.inv(model.precisions), f'{args.classifier_type}: initial')

if args.is_toy:
    plot_decision_regions(model.prototypes, test_features_subset, test_labels_subset,os.path.join(args.root, "initial_test.pdf"), plot_config, model, device)
    if args.noisy_target:
        confusing_logits = model.predict(confusing_test_features)
        print_accuracy(confusing_logits, confusing_labels, f'{args.classifier_type}, target accuracy confusing')
        easy_logits = model.predict(easy_test_features)
        print_accuracy(easy_logits, easy_labels, f'{args.classifier_type}, target accuracy easy')

#train_logistic_regression_head(train_features, train_labels, test_features, test_labels)

# Flip label experiment
flipped_train_labels = train_labels.clone()
indices_to_flip = torch.randperm(len(train_labels))[0:int(len(train_labels)*args.flip_fraction)]
flip_offset = torch.from_numpy(rand.randint(low=1, high=num_classes, size=(len(indices_to_flip)))).type(torch.LongTensor).to(device)
flipped_train_labels[indices_to_flip] = (flipped_train_labels[indices_to_flip] + flip_offset) % num_classes
#print("Flipped sum: {}".format(flipped_train_labels.sum()))

#flipped_test_labels = test_labels.clone()
#test_indices_to_flip = torch.randperm(len(test_labels))[0:int(len(test_labels)*args.flip_fraction)]
#flipped_test_labels[test_indices_to_flip] = 1 - flipped_test_labels[test_indices_to_flip]
logits = model(train_features, flipped_train_labels, test_features)
print_accuracy(logits, test_labels, f'{args.classifier_type}: {args.flip_fraction*100}% Noisy')
if args.classifier_type == 'Protonets':
    print_prototype_info(euclidean_metric(model.prototypes, model.prototypes), model.stds, f'{args.classifier_type}: Noisy')
else:
    print_prototype_info(euclidean_metric(model.prototypes, model.prototypes), torch.linalg.inv(model.precisions), f'{args.classifier_type}: Noisy')


if args.is_toy:
    plot_decision_regions(model.prototypes, test_features_subset, test_labels_subset, os.path.join(args.root, "noisy_{}.pdf").format(args.flip_fraction*100), plot_config, model, device)

#train_logistic_regression_head(train_features, flipped_train_labels, test_features, test_labels)


if args.ranking_method == 'loo':
    rankings = calculate_rankings(model, train_features, flipped_train_labels, test_features, test_labels, num_classes)
elif args.ranking_method == 'random':
    rankings = calculate_random_rankings(flipped_train_labels)
#rankings = calculate_rankings(model, train_features, flipped_train_labels, test_features, flipped_test_labels, num_classes)

num_correctly_identified = []
relabelled_model_acc = []
relabelled_model_loss = []
relabelled_lreg_acc = []

for check_fraction in check_fractions:
    num_to_keep = int(len(train_features) * (1 - check_fraction ))
    keep_indices = rankings[0:num_to_keep]
    relabel_indices = rankings[num_to_keep:]
    num_correct_indices = len(set(indices_to_flip.numpy()).intersection(set(relabel_indices.numpy())))
    num_correctly_identified.append(num_correct_indices)
    print(f'Correctly identified indices {(num_correct_indices)} out of {(args.flip_fraction*len(train_features))}')

    relabeled_train_labels = flipped_train_labels.clone()
    relabeled_train_labels[relabel_indices] = train_labels[relabel_indices]

    logits = model(train_features, relabeled_train_labels, test_features)
    acc, loss = print_accuracy(logits, test_labels, f'{args.classifier_type}: {check_fraction*100}% relabeled', hush=False)
    if args.classifier_type == 'Protonets':
        print_prototype_info(euclidean_metric(model.prototypes, model.prototypes), model.stds, f'{args.classifier_type}: {check_fraction*100}% relabeled')
    else:
        print_prototype_info(euclidean_metric(model.prototypes, model.prototypes), torch.linalg.inv(model.precisions), f'{args.classifier_type}: {check_fraction*100}% relabeled')

    relabelled_model_acc.append(acc)
    relabelled_model_loss.append(loss.item())

    #test_acc = train_logistic_regression_head(train_features, relabeled_train_labels, test_features, test_labels)
    #relabelled_lreg_acc.append(test_acc)

logfile.write("Check fractions: {}\n".format(check_fractions))
logfile.write("Num correctly identified: {}\n".format(num_correctly_identified))
logfile.write("Relabelled {} acc: {}\n".format(args.classifier_type, relabelled_model_acc))
logfile.write("Relabelled {} loss: {}\n".format(args.classifier_type, relabelled_model_loss))
logfile.write("Relabelled LReg acc: {}\n".format(relabelled_lreg_acc))
