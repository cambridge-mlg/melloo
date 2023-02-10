def get_dataset_reader(args, train_set, validation_set, test_set, device, value_tracking=False):
    query_test = args.query_test
    if args.indep_eval:
        query_test *= 2
    if args.dataset == "meta-dataset" or args.dataset == "meta-dataset_ilsvrc_only":
        if args.dataset_reader == "official":
            from meta_dataset_reader import MetaDatasetReader
            dataset = MetaDatasetReader(args.data_path, args.mode, train_set, validation_set, test_set,
                                        args.max_way_train, args.max_way_test, args.max_support_train,
                                        args.max_support_test, args.query_train, query_test, args.image_size)
        else:
            from meta_dataset_reader_pytorch import MetaDatasetReaderPyTorch
            dataset = MetaDatasetReaderPyTorch(args.data_path, args.mode, train_set, validation_set, test_set,
                                               args.max_way_train, args.max_way_test, args.max_support_train,
                                               args.max_support_test, args.query_train, query_test,
                                               args.image_size, device)
    elif args.dataset == "split-cifar10" or args.dataset == "split-mnist":
        if value_tracking:
            from split_wrapper import ValueTrackingDatasetWrapper
            dataset = ValueTrackingDatasetWrapper(args.data_path, args.dataset, args.way, args.shot, query_test)
        else:
            from split_wrapper import IdentifiableDatasetWrapper
            dataset = IdentifiableDatasetWrapper(args.data_path, args.dataset, args.way, args.shot, query_test)
    else:
        from meta_dataset_reader import SingleDatasetReader
        dataset = SingleDatasetReader(args.data_path, args.mode, args.dataset, args.way, args.shot, args.query_train,
                                      query_test, args.image_size)

    return dataset
