import torch
import torchvision as tv
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image
#import utils

def dataset_from_metdataset_task_dict(task_dict):
    return {train_set: {train_data: task_dict['context_images'], train_labels: task_dict['context_labels']},
                test_set: {test_data: task_dict['target_images'], test_labels: task_dict['target_labels']}}

class CIFAR(tv.datasets.CIFAR10):
    """Wrapper around the MNIST dataset to ensure compatibility with our
    implementation.
    """

    def __init__(self, way, train_shot, test_shot, dataset=None):
        if dataset is None:
            #tr = transforms.Compose([transforms.Resize((84, 84)), transforms.ToTensor()])
            self.train_set = tv.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=None)
            self.test_set = tv.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=None)
            self.test_set.test_data = self.preprocess_images(self.test_set.test_data)
            self.train_set.train_data = self.preprocess_images(self.train_set.train_data)
        else:
            self.train_set = dataset.train_set
            self.test_set = dataset.test_set
        
        self.way = way
        self.train_shot = train_shot
        self.test_shot = test_shot

        self.train_indices_by_class = self.get_indices_by_class(self.train_set.train_labels)
        self.test_indices_by_class = self.get_indices_by_class(self.test_set.test_labels)

    def preprocess_images(self, images):
        result_images = np.ndarray((len(images), 84,84,3), dtype="float32")
        for i in range(len(images)):
            pil_image = Image.fromarray(images[i])
            resized_image = pil_image.resize((84,84), Image.BICUBIC)
            result_images[i] = np.array(resized_image)
        result_images = (result_images / 255.0)*2.0 - 1

        return result_images

    def get_indices_by_class(self, labels):
        indices_by_class = {}
        class_labels = np.unique(labels)
        for i in class_labels:
            indices_by_class[i] = []
        for i in range(len(labels)):
            label = labels[i]
            indices_by_class[label].append(i)
        return indices_by_class
    
    def shuffle_dictionary(self, dict):
        shuffled_dict = dict.copy()
        for key in shuffled_dict.keys():
            random.shuffle(shuffled_dict[key])

        return shuffled_dict

    def get_train_task(self):
        return
        
    def get_test_task(self):
        return
        
    def get_covering_tasks(self):
        # CIFAR10 has 60000 images voer 10 classes, of which 50000 are training images
        # We thus have 5000 training images per class.
        # So we're going to divie the 5000 training images up into 1000 shot-sets
        # And then we're going to form random tasks using those 1000 shot-sets

        class_labels = np.unique(self.train_set.train_labels)
        shuffled_indices_by_class = self.shuffle_dictionary(self.train_indices_by_class)

        num_tasks = len(self.train_set.train_data)/(self.train_shot * self.way)
        tasks = []
        
        while len(tasks) < num_tasks:
            available_classes = class_labels.copy().tolist()
            for k in range(int(len(class_labels)/self.way)):
                selected_classes = random.sample(available_classes, self.way)
                for c in selected_classes:
                    available_classes.remove(c)
                images, labels, indices = [], [], []
                test_images, test_labels, test_indices = [], [], []
                for ic, c in enumerate(selected_classes):
                    image_indices = shuffled_indices_by_class[c][0:self.train_shot]
                    for im in image_indices:
                        images.append(self.train_set.train_data[im])

                    indices.extend(image_indices)
                    shuffled_indices_by_class[c] = shuffled_indices_by_class[c][self.train_shot:]
                    labels.extend([ic]*self.train_shot)

                    test_image_indices = random.sample(self.test_indices_by_class[c], self.test_shot)
                    for im in test_image_indices:
                        test_images.append(self.test_set.test_data[im])

                    test_indices.extend(test_image_indices)
                    test_labels.extend([ic]*self.test_shot)

                images = np.array(images, dtype="float32")
                #images = (images / 255.0)*2.0 - 1

                test_images = np.array(test_images, dtype="float32")
                #test_images = (test_images / 255.0)*2.0 - 1

                labels = np.array(labels, dtype="int32")
                test_labels = np.array(test_labels, dtype="int32")

                tasks.append({'context_images': images,
                              'context_labels': labels,
                              'context_indices': indices,
                              'target_images': test_images,
                              'target_labels': test_labels,
                              'target_indices': test_indices})

        return tasks
        
    def get_covering_test_tasks(self):
        return
    
    
def valid_task_cover(tasks, class_labels, shot, way):
    label_sums = {}
    for c in class_labels:
        label_sums[c] = 0
    for task in tasks:
        for l in task['context_labels']:
            label_sums[l] = label_sums[l] + 1
    for c in class_labels:
        assert(label_sums[c] == 5000)
    print("PASSED: Correct label counts")
    unique_indices = set([])
    all_indices = []
    for task in tasks:
        unique_indices.update(task['context_indices'])
        all_indices.extend(task['context_indices'])
    assert len(unique_indices) == len(all_indices)
    print("PASSED: All indices unique")

    for task in tasks:
        assert len(task['context_images']) == shot * way
        assert len(task['context_labels']) == shot* way
        assert len(np.unique(task['context_labels'])) == way
        assert len(task['context_indices']) == shot* way
    print("PASSED: Correct number of items according to specified shot and way")

if __name__ == "__main__":
    cifar = CIFAR(5, 5, 5)
    tasks = cifar.get_covering_tasks()
    valid_task_cover(tasks, list(range(0, 10)), 5, 5)
