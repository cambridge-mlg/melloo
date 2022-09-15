ulimit -n 50000
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=4

python3 ./learners/protonets_attention/src/main.py \
    --classifier protonets_mahalanobis \
    --mode test \
    --data_path /scratch/etv21/debug_discard \
    --dataset split-cifar10 \
    --test_datasets split-cifar10 \
    --dataset_reader pytorch \
    --checkpoint /scratch/etv21/debug \
    --test_model_path /scratches/stroustrup/etv21/protonets_attention/checkpoints_l2/fully_trained.pt \
    --pretrained_resnet_path learners/protonets_attention/models/pretrained_resnet.pt.tar \
    --batch_normalization basic \
    --feature_adaptation film \
    --way 10 \
    --shot 5 \
    --query_test 10 \
    --l2_regularize_classifier \
    --top_k 1 \
    --selection_mode drop \
    --importance_mode kl_loo \
    --kernel_agg class  \
    --tasks 500 \
    --spread_constraint nonempty \
    --task_type generate_coreset_discard

# loo_monotonic_10_way_5_shot_drop_1 \
#    --dataset cifar10 \
#    --data_path /scratches/stroustrup/jfb54/tf-meta-dataset/records \
