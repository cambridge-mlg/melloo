ulimit -n 50000
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0


python3 ./learners/protonets_attention/src/main.py \
    --classifier protonets_attention \
    --mode test \
    --test_datasets ilsvrc_2012 \
    --data_path /scratches/stroustrup/jfb54/tf-meta-dataset/records \
    --test_model_path /scratches/stroustrup/etv21/protonets_attention/checkpoints_l2/fully_trained.pt \
    --pretrained_resnet_path learners/protonets_attention/models/pretrained_resnet.pt.tar \
    --batch_normalization basic \
    --feature_adaptation film \
    --max_support_test 50 \
    --max_way_test 5 \
    --dataset meta-dataset_ilsvrc_only \
    --l2_regularize_classifier

