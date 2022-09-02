import numpy as np
from numpy.random import default_rng

import torch
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms
from PIL import Image

rng = default_rng()
shorten_data = False
shortened_class_size = 10

def convert_to_array(my_list):
    if type(my_list) == np.int32 or type(my_list) == int:
        return np.array([my_list])
    return my_list


def map_to_classes(dataset):
    class_mapping = []
    for c in range(len(dataset.classes)):
        class_mapping.append([])
    for index, (img, label) in enumerate(dataset):
        class_mapping[label].append(index)
    if shorten_data:
        for c in range(len(dataset.classes)):
            class_mapping[c] = class_mapping[c][0:shortened_class_size]
    return class_mapping

def reset_empty_class_mapings(dataset, class_mapping, shot):
    # Make a list of all the classes that have been exhausted
    empty_classes = []
    for c in range(len(class_mapping)):
        if len(class_mapping[c]) < shot:
            empty_classes.append(c)
            # Make it a list containing the unused patterns
            # (so they have a bigger chance of being sampled next round)
            class_mapping[c] = class_mapping[c].tolist()

    # If no classes have been exhausted, return mapping unchanged
    if len(empty_classes) == 0:
        return class_mapping

    for index, (img, label) in enumerate(dataset):
        if label in empty_classes:
            class_mapping[label].append(index)

    if shorten_data:
        for c in range(len(class_mapping)):
            class_mapping[c] = class_mapping[c][0:shortened_class_size]

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

        if dataset_name == "split-cifar10":
            transforms = tv_transforms.Compose([
                tv_transforms.Resize(84, interpolation=Image.LANCZOS),
                tv_transforms.ToTensor(),
                ])

            self.context_data = tv_datasets.CIFAR10(dataset_path, transform=transforms, train=True, download=True)
            self.query_data = tv_datasets.CIFAR10(dataset_path, transform=transforms, train=False, download=True)
        elif dataset_name == "split-mnist":
            transforms = tv_transforms.Compose([
                    tv_transforms.Resize(84, interpolation=Image.LANCZOS),
                    tv_transforms.ToTensor(),
                    tv_transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                ])

            self.context_data = tv_datasets.MNIST(dataset_path, transform=transforms, train=True, download=True)
            self.query_data = tv_datasets.MNIST(dataset_path, transform=transforms, train=False, download=True)
        else:
            print("Unsupported dataset specified: {}".format(dataset_name))
        '''
        nc = 10
        ns = 100
        self.context_data.classes = self.context_data.classes[0:nc]
        self.context_data.data = self.context_data.data[0:ns]
        self.context_data.targets = self.context_data.targets[0:ns]
        for i in range(ns):
            self.context_data.targets[i] = i % nc
        self.query_data.classes = self.query_data.classes[0:nc]
        self.query_data.data = self.query_data.data[0:ns]
        self.query_data.targets = self.query_data.targets[0:ns]
        for i in range(ns):
            self.query_data.targets[i] = i % nc
        '''

        self.query_mapping = map_to_classes(self.query_data)
        # Now we want splits for this per class so we can construct tasks
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

    def get_total_num_classes(self):
        num_context_classes = len(self.context_data.classes)
        assert num_context_classes == len(self.query_data.classes)
        return num_context_classes

    def get_validation_task(self, *args):
        # Not implemented
        return None

    def get_train_task(self, *args):
        # Not implemented
        return None

    # By default, remove issued points from the context set mapping.
    # but leave points in the query set alone so they can be re-issued later. The query point behaviour is what we used in the older experiments, so we've kept the deault the same.
    def get_test_task(self, *args, mutate_context_mapping=True, mutate_query_mapping=False):
        if mutate_context_mapping:
            self.current_context_mapping = reset_empty_class_mapings(self.context_data, self.current_context_mapping, self.shot)

        c_avail = available_classes(self.current_context_mapping)

        # No data left, reset
        #if len(c_avail) == 0:
        #    self.current_context_mapping = map_to_classes(self.context_data)
        #    c_avail = available_classes(self.current_context_mapping)


        task_dict = {}
        task_dict["context_images"], unnormalized_context_labels, task_dict["context_ids"] = self._construct_context_set(c_avail, mutate_context_mapping=mutate_context_mapping)
        # Use our custom, order-preserving unique function
        chosen_classes = unique(unnormalized_context_labels)
        task_dict["target_images"], unnormalized_target_labels, task_dict["target_ids"] = self._construct_query_set(chosen_classes, mutate_query_mapping=mutate_query_mapping)

        # If all classes are represented, don't renormalize
        if len(chosen_classes) == len(self.context_data.classes):
            task_dict["context_labels"], task_dict["target_labels"] = unnormalized_context_labels, unnormalized_target_labels
        else:
            # Here we make the asssumption that all classes are represented equally in the context and target sets (but the whole setup assumes that)
            norm_labels = torch.arange(len(chosen_classes))
            task_dict["context_labels"], task_dict["target_labels"] = norm_labels.repeat_interleave(self.shot), norm_labels.repeat_interleave(self.query_shot)

        return task_dict
    
    def get_labels_from_ids(self, context_image_ids):
        num_context = len(context_image_ids)
        context_labels = torch.zeros(num_context, dtype=torch.long)
        for i, id in enumerate(context_image_ids):
            _, context_labels[i] = self.context_data[id]
        return context_labels


    # Note how this doesn't renormalize class labels
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

    def _construct_query_set(self, task_classes, mutate_query_mapping=False):
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
            if mutate_query_mapping:
                self.query_mapping[c] = np.delete(self.query_mapping[c], pattern_indices)
        return task_images, task_labels, task_ids

    # By default, we always mutate the context mapping. So by default, we won't return the same point until we've run out of all other points.
    def _construct_context_set(self, possible_classes, mutate_context_mapping=True):
        # Choose the classes for the task
        try:
            task_classes = rng.choice(possible_classes, size=self.way, replace=False)
        except ValueError:
            print("Something bad happened when sampling classes")
            import pdb; pdb.set_trace()

        task_images = torch.zeros(self._context_task_shape(), dtype=torch.float32)
        task_labels = torch.zeros(self.shot*self.way, dtype=torch.long)
        task_ids = np.zeros(self.shot*self.way, dtype=np.int32)
        t = 0
        for c in task_classes:
            c_indices = self.current_context_mapping[c]
            # From that class, choose (and remove) available instances
            try:
                pattern_indices = rng.choice(len(c_indices), self.shot, replace=False)
            except ValueError:
                print("Something bad happened when sampling indices from {}".format(c_indices))
                import pdb; pdb.set_trace()
            for i in pattern_indices:
                pattern, label = self.context_data[c_indices[i]]
                assert label == c
                task_images[t] = pattern
                task_labels[t] = label
                task_ids[t] = c_indices[i]
                t += 1
            # Remove selected patterns from list of available patterns to choose from
            if mutate_context_mapping:
                self.current_context_mapping[c] = np.delete(self.current_context_mapping[c], pattern_indices)
        return task_images, task_labels, task_ids


### This class maintains a list of the context points currently issued so that they are not re-issued
### It also tracks how many "rounds" a context point stays issued for, under the assumption that good context points
### won't be discarded as often as bad ones.
### It uses the IdentifiableDatasetWrapper for some functionality, like constructing query sets.
### Use of this class:
### Use the get_test_task function to obtain the starting task.
###     These context points are no longer drawable. But they will remain in the underlying IdentifiableDatasetWrapper's mapping, 
###     which we try to leacve unchanged throuhgout, reyling instead on the list of drawable and current context ids.'
### You can request new context points using sample_new_context_points after calling mark_discarded to discard the relevant points.
###     Marking a point discarded will take it out of the list of currently issued context points. 
###     They will stay out of rotation until the next call to reset_drawable_ids, which will happen when more new context points are requested than are available in the list of drawable_context_ids.
### 
### You can request a new queryset, which will rely on the underlying IdentifiableDataset to retrieve a queryset from the set of available query points.
###     These points will still be available to draw next time you ask for a query set.
###     To prevent the initial seed query set from being re-issued, the call to get_test_task is done with mutate_query_mapping=True so that those points can never be re-issued when we're requesting
###     evaluation points.

class ValueTrackingDatasetWrapper(IdentifiableDatasetWrapper):
    def __init__(self, dataset_path, dataset_name, way, shot, query_shot):
        IdentifiableDatasetWrapper.__init__(self, dataset_path, dataset_name, way, shot, query_shot)

        # Construct list of drawable ids from current_context_mapping
        self.current_context_ids = []
        self.drawable_context_ids = self._reset_drawable_ids()
        self.rounds_not_discarded = {}
        self.returned_label_counts = {}
        # TODO: Recomputing this every time is expensive, save a master version and make a copy on reset.
        self.master_drawable_list = []

    def _reset_drawable_ids(self):
        drawable_ids = []
        for cl in range(len(self.context_data.classes)):
            drawable_ids = drawable_ids + self.current_context_mapping[cl]
        
        # If some points are currently issued, remove them from the drawable list
        if len(self.current_context_ids) > 0:
            for context_id in self.current_context_ids:
                drawable_ids.remove(context_id)

        return drawable_ids


    # For the ValueTracking dataset wrapper, the requested way should match actual classes
    def get_test_task(self, *args):
        assert self.way == len(self.context_data.classes)
        # Here we choose not to mutate the context mapping, we rely on the drawable id list instead.
        task_dict = IdentifiableDatasetWrapper.get_test_task(self, *args, mutate_context_mapping=False, mutate_query_mapping=True)
        # Track what images are currently in rotation
        self.current_context_ids = task_dict["context_ids"].tolist()
        # Mark the selected context images so that we don't swap them in multiple times
        for i, context_id in enumerate(self.current_context_ids):
            assert context_id in self.drawable_context_ids
            self.drawable_context_ids.remove(context_id)
            if task_dict["context_labels"][i].item() in self.returned_label_counts:
                self.returned_label_counts[task_dict["context_labels"][i].item()] += 1
            else:
                self.returned_label_counts[task_dict["context_labels"][i].item()] = 1
        return task_dict

    def mark_discarded(self, image_ids):
        image_ids = convert_to_array(image_ids)
        # Increase count for images not discarded this round
        current_context_set = set(self.current_context_ids)
        not_discarded_ids_set = current_context_set.difference(set(image_ids))
        if len(not_discarded_ids_set) > 0:
            not_discarded_ids = list(current_context_set.difference(set(image_ids)))
            for nd_id in not_discarded_ids:
                if nd_id in self.rounds_not_discarded.keys():
                    self.rounds_not_discarded[nd_id] += 1
                else:
                    self.rounds_not_discarded[nd_id] = 1
        try:
            # Remove discarded from context set
            for image_id in image_ids:
                self.current_context_ids.remove(image_id)
        except ValueError:
            import pdb; pdb.set_trace()
    
    def _calculate_available_class_distrib(self):
        class_counts = {}
        for id in self.drawable_context_ids:
            _, label = self.context_data[id]
            #label = label.item()
            if label in class_counts.keys():
                class_counts[label] += 1
            else:
                class_counts[label] = 1

        return class_counts
            

    def sample_new_context_points(self, num_points_requested, force_reset=False):
        # Check whether there are new points to propose
        # If not, reset the list of drawables, excluding current selection
        if len(self.drawable_context_ids) < num_points_requested or force_reset:
            self.drawable_context_ids = self._reset_drawable_ids()
            class_distribs = self._calculate_available_class_distrib()
            print(class_distribs)


        #import pdb; pdb.set_trace()

        new_ids = rng.choice(self.drawable_context_ids, size=num_points_requested, replace=False)
        for id in new_ids:
            self.drawable_context_ids.remove(id)
        self.current_context_ids = self.current_context_ids + new_ids.tolist()

        context_images = torch.zeros(self._context_task_shape(num_points_requested))
        context_labels = torch.zeros(num_points_requested, dtype=torch.long)
        for i, id in enumerate(new_ids):
            context_images[i], context_labels[i] = self.context_data[id]
            if context_labels[i].item() in self.returned_label_counts:
                self.returned_label_counts[context_labels[i].item()] += 1
            else:
                self.returned_label_counts[context_labels[i].item()] = 1
        #print("ValueTracker has {} points issued".format(len(self.current_context_ids)))
        return context_images, context_labels, new_ids

    def get_query_set(self):
        return self._construct_query_set(range(len(self.query_data.classes)))
