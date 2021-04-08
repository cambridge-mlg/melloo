import torch
from pytorch_meta_dataset.utils import Split
import pytorch_meta_dataset.config as config_lib
import pytorch_meta_dataset.dataset_spec as dataset_spec_lib
from torch.utils.data import DataLoader
import os
import pytorch_meta_dataset.pipeline as pipeline
import numpy as np


def shuffle(images, labels):
    """
    Return shuffled data.
    """
    permutation = np.random.permutation(images.shape[0])
    return images[permutation], labels[permutation]


class MetaDatasetReaderPyTorch:
    def __init__(self, data_path, mode, train_set, validation_set, test_set, max_way_train, max_way_test,
                 max_support_train, max_support_test, max_query_train, max_query_test, image_size, device):
        self.device = device
        self.train_it = None
        self.validation_it_dict = {}
        self.test_it_dict = {}

        if mode == 'train' or mode == 'train_test':
            train_data_config = config_lib.DataConfig(
                sources=train_set,
                data_path=data_path,
                image_size=image_size
            )
            train_episode_config = config_lib.EpisodeDescriptionConfig(
                num_ways=None,
                num_support=None,
                num_query=None,
                max_ways_upper_bound=max_way_train,
                max_num_query=max_query_train,
                max_support_set_size=max_support_train
            )

            self.train_it = self._init_episodic_dataset(datasets=train_set,
                                                        split=Split["TRAIN"],
                                                        data_config=train_data_config,
                                                        episode_config=train_episode_config)

            validation_data_config = config_lib.DataConfig(
                sources=validation_set,
                data_path=data_path,
                image_size=image_size
            )
            validation_episode_config = config_lib.EpisodeDescriptionConfig(
                num_ways=None,
                num_support=None,
                num_query=None,
                max_ways_upper_bound=max_way_test,
                max_num_query=max_query_test,
                max_support_set_size=max_support_test
            )
            for item in validation_set:
                it = self._init_episodic_dataset(datasets=[item],
                                                 split=Split["VALID"],
                                                 data_config=validation_data_config,
                                                 episode_config=validation_episode_config)
                self.validation_it_dict[item] = it

        if mode == 'test' or mode == 'train_test':
            test_data_config = config_lib.DataConfig(
                sources=test_set,
                data_path=data_path,
                image_size=image_size
            )

            test_episode_config = config_lib.EpisodeDescriptionConfig(
                num_ways=None,
                num_support=None,
                num_query=None,
                max_ways_upper_bound=max_way_test,
                max_num_query=max_query_test,
                max_support_set_size=max_support_test
            )

            for item in test_set:
                it = self._init_episodic_dataset(datasets=[item],
                                                 split=Split["TEST"],
                                                 data_config=test_data_config,
                                                 episode_config=test_episode_config)
                self.test_it_dict[item] = it

    def _init_episodic_dataset(self, datasets, split, data_config, episode_config):
        # Get the data specifications
        use_bilevel_ontology_list = [False]*len(datasets)
        use_dag_ontology_list = [False]*len(datasets)

        # Enable ontology aware sampling for Omniglot and ImageNet.
        if 'omniglot' in datasets:
            use_bilevel_ontology_list[datasets.index('omniglot')] = True
        if 'imagenet' in datasets:
            use_dag_ontology_list[datasets.index('ilsvrc_2012')] = True
        episode_config.use_bilevel_ontology_list = use_bilevel_ontology_list
        episode_config.use_dag_ontology_list = use_dag_ontology_list

        all_dataset_specs = []
        for dataset_name in datasets:
            dataset_records_path = os.path.join(data_config.path, dataset_name)
            dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
            all_dataset_specs.append(dataset_spec)

        # Form an episodic dataset
        episodic_dataset = pipeline.make_episode_pipeline(dataset_spec_list=all_dataset_specs,
                                                          split=split,
                                                          data_config=data_config,
                                                          episode_descr_config=episode_config)

        # Use a standard dataloader
        episodic_loader = DataLoader(dataset=episodic_dataset,
                                     batch_size=1,
                                     num_workers=data_config.num_workers)

        return iter(episodic_loader)

    def _get_task(self, it):
        (support, query, support_labels, query_labels) = next(it)

        # shuffle
        support, support_labels = shuffle(support.squeeze(dim=0), support_labels.squeeze(dim=0))
        query, query_labels = shuffle(query.squeeze(dim=0), query_labels.squeeze(dim=0))

        support, support_labels = support.to(self.device), support_labels.to(self.device, non_blocking=True)
        query, query_labels = query.to(self.device), query_labels.to(self.device, non_blocking=True)
        task_dict = {
            'context_images': support,
            'context_labels': support_labels,
            'target_images': query,
            'target_labels': query_labels
        }
        return task_dict

    def get_train_task(self):
        return self._get_task(self.train_it)

    def get_validation_task(self, item):
        return self._get_task(self.validation_it_dict[item])

    def get_test_task(self, item):
        return self._get_task(self.test_it_dict[item])


class SingleDatasetReader:
    """
    Class that wraps the Meta-Dataset episode reader to read in a single dataset.
    """
    def __init__(self, data_path, mode, dataset, way, shot, query_train, query_test, image_size):
        self.data_path = data_path
        self.train_next_task = None
        self.validation_next_task = None
        self.test_next_task = None
        self.image_size = image_size


        fixed_way_shot_train = self._get_train_episode_description(num_ways=way, num_support=shot, num_query=query_train)
        fixed_way_shot_test = self._get_test_episode_description(num_ways=way, num_support=shot, num_query=query_test)

        if mode == 'train' or mode == 'train_test':
            self.train_next_task = self._init_dataset(dataset, learning_spec.Split.TRAIN, fixed_way_shot_train)
            self.validation_next_task = self._init_dataset(dataset, learning_spec.Split.VALID, fixed_way_shot_test)

        if mode == 'test' or mode == 'train_test' or mode == 'onnx':
            self.test_next_task = self._init_dataset(dataset, learning_spec.Split.TEST, fixed_way_shot_test)

    def _init_dataset(self, dataset, split, episode_description):
        dataset_records_path = os.path.join(self.data_path, dataset)
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)

        single_source_pipeline = pipeline.make_one_source_episode_pipeline(
            dataset_spec=dataset_spec,
            use_dag_ontology=False,
            use_bilevel_ontology=False,
            split=split,
            episode_descr_config=episode_description,
            image_size=self.image_size)

        iterator = single_source_pipeline.make_one_shot_iterator()
        return iterator.get_next()

    def _get_task(self, next_task):
        (episode, source_id) = self.session.run(next_task)
        task_dict = {
            'context_images': episode[0],
            'context_labels': episode[1],
            'target_images': episode[3],
            'target_labels': episode[4]
        }
        return task_dict

    def get_train_task(self):
        return self._get_task(self.train_next_task)

    def get_validation_task(self, item):
        return self._get_task(self.validation_next_task)

    def get_test_task(self, item):
        return self._get_task(self.test_next_task)


    def _get_train_episode_description(self, num_ways, num_support, num_query):
        return config.EpisodeDescriptionConfig(
            num_ways=num_ways,
            num_support=num_support,
            num_query=num_query,
            min_ways=5,
            max_ways_upper_bound=50,
            max_num_query=10,
            max_support_set_size=500,
            max_support_size_contrib_per_class=100,
            min_log_weight=-0.69314718055994529, # np.cnaps_layer_log.txt(0.5)
            max_log_weight=0.69314718055994529, # np.cnaps_layer_log.txt(2)
            ignore_dag_ontology=False,
            ignore_bilevel_ontology=False,
            ignore_hierarchy_probability=0.0,
            simclr_episode_fraction=0.0
        )

    def _get_test_episode_description(self, num_ways, num_support, num_query):
        return config.EpisodeDescriptionConfig(
            num_ways=num_ways,
            num_support=num_support,
            num_query=num_query,
            min_ways=5,
            max_ways_upper_bound=50,
            max_num_query=10,
            max_support_set_size=500,
            max_support_size_contrib_per_class=100,
            min_log_weight=-0.69314718055994529, # np.cnaps_layer_log.txt(0.5)
            max_log_weight=0.69314718055994529, # np.cnaps_layer_log.txt(2)
            ignore_dag_ontology=False,
            ignore_bilevel_ontology=False,
            ignore_hierarchy_probability=0.0,
            simclr_episode_fraction=0.0
        )