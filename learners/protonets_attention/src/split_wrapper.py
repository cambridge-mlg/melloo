import numpy as np
from numpy.random import default_rng

import torch
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms
from PIL import Image

rng = default_rng()

def map_to_classes(dataset):
    class_mapping = []
    for c in range(len(dataset.classes)):
        class_mapping.append([])
    for index, (img, label) in enumerate(dataset):
        class_mapping[label].append(index)
    return class_mapping
    
def available_classes(class_mapping):
    c_avail = []
    for c, c_indices in enumerate(class_mapping):
        if len(c_indices) > 0:
            c_avail.append(c)
    return c_avail

def unique(values):
    result = []
    for v in values:
        if v not in result:
            result.append(v.item())
    return result

class IdentifiableDatasetWrapper:
    def __init__(self, dataset_path, dataset_name, way, shot, query_shot):
        transforms = tv_transforms.Compose([
                tv_transforms.Resize(84, interpolation=Image.LANCZOS),
                tv_transforms.ToTensor(),
            ])
        self.context_data = tv_datasets.CIFAR10(dataset_path, transform=transforms, train=True, download=True)
        #self.context_data.data = self.context_data.data[0:30]
        #for i in range(30):
        #    self.context_data.targets[i] = i % 5
        self.query_data = tv_datasets.CIFAR10(dataset_path, transform=transforms, train=False, download=True)
        self.query_mapping = map_to_classes(self.query_data)
        # Now we want splits for this per class so we can construct tasks
        #self.class_mapping = map_to_classes(self.dataset)
        # This is the version that we'll be editing
        self.current_context_mapping = map_to_classes(self.context_data)
        self.way = way
        self.shot = shot
        self.query_shot = query_shot
        
    def _context_task_shape(self, task_size=None):
        if task_size is None:
            task_size = self.way * self.shot
        image_shape = self.context_data[0][0].shape
        return (task_size, image_shape[0], image_shape[1], image_shape[2])
        
    def _query_task_shape(self, task_size=None):
        if task_size is None:
            task_size = self.way * self.query_shot
        image_shape = self.query_data[0][0].shape
        return (task_size, image_shape[0], image_shape[1], image_shape[2])
        
    def get_validation_task(self, *args):
        # Not implemented
        return None
        
    def get_train_task(self, *args):
        # Not implemented
        return None
        
    def get_test_task(self, *args):
        c_avail = available_classes(self.current_context_mapping)
        
        # No data left, reset
        if len(c_avail) == 0:
            self.current_context_mapping = map_to_classes(self.context_data)
            c_avail = available_classes(self.current_context_mapping)
        
        task_dict = {}
        task_dict["context_images"], unnormalized_context_labels, task_dict["context_ids"] = self._construct_context_set(c_avail)
        # Use our custom, order-preserving unique function
        chosen_classes = unique(unnormalized_context_labels)
        task_dict["target_images"], unnormalized_target_labels, task_dict["target_ids"] = self._construct_query_set(chosen_classes)
        # Here we make the asssumption that all classes are represented equally in the context and target sets (but the whole setup assumes that)
        norm_labels = torch.arange(len(chosen_classes))
        task_dict["context_labels"], task_dict["target_labels"] = norm_labels.repeat_interleave(self.shot), norm_labels.repeat_interleave(self.query_shot)

        return task_dict
        
    def get_task_from_ids(self, context_image_ids):
        task_dict = {}
        num_context = len(context_image_ids)
        context_images = torch.zeros(self._context_task_shape(num_context))
        context_labels = torch.zeros(num_context, dtype=torch.long)
        for i, id in enumerate(context_image_ids):
            context_images[i], context_labels[i] = self.context_data[id]
        task_dict["context_images"], task_dict["context_labels"] = context_images, context_labels
        full_test_classes = list(range(len(self.context_data.classes)))
        task_dict["target_images"], task_dict["target_labels"], _ = self._construct_query_set(full_test_classes)
        return task_dict
        
    def _construct_query_set(self, task_classes):   
        task_way = len(task_classes)
        task_images = torch.zeros(self._query_task_shape(task_way*self.query_shot), dtype=torch.float32)
        task_labels = torch.zeros(self.query_shot*task_way, dtype=torch.long)
        task_ids = np.zeros(self.query_shot*task_way, dtype=np.int32)
        t = 0
        for c in task_classes:
            c_indices = self.query_mapping[c]
            # From that class, choose (and remove) available instances
            pattern_indices = rng.choice(len(c_indices), self.query_shot, replace=False)
            for i in pattern_indices:
                pattern, label = self.query_data[c_indices[i]]
                assert label == c
                task_images[t] = pattern
                task_labels[t] = label
                task_ids[t] = c_indices[i]
                t += 1
        return task_images, task_labels, task_ids
        
    def _construct_context_set(self, possible_classes):
        # Choose the classes for the task
        task_classes = rng.choice(possible_classes, size=self.way, replace=False)

        task_images = torch.zeros(self._context_task_shape(), dtype=torch.float32)
        task_labels = torch.zeros(self.shot*self.way, dtype=torch.long)
        task_ids = np.zeros(self.shot*self.way, dtype=np.int32)
        t = 0
        for c in task_classes:
            c_indices = self.current_context_mapping[c]
            # From that class, choose (and remove) available instances
            pattern_indices = rng.choice(len(c_indices), self.shot, replace=False)
            for i in pattern_indices:
                pattern, label = self.context_data[c_indices[i]]
                assert label == c
                task_images[t] = pattern
                task_labels[t] = label
                task_ids[t] = c_indices[i]
                t += 1
            # Remove selected patterns from list of available patterns to choose from
            self.current_context_mapping[c] = np.delete(self.current_context_mapping[c], pattern_indices) 
        return task_images, task_labels, task_ids
                
            
        
        
        
