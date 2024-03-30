ulimit -n 50000
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1


python3 ./learners/protonets_attention/src/main.py \
    --classifier protonets_attention \
    --mode test \
    --data_path /scratches/stroustrup/etv21/ \
    --dataset split-mnist \
    --test_datasets split-mnist \
    --dataset_reader pytorch \
    --checkpoint /scratches/stroustrup/etv21/debug \
    --test_model_path /scratches/stroustrup/etv21/protonets_attention/checkpoints_l2/fully_trained.pt \
    --pretrained_resnet_path learners/protonets_attention/models/pretrained_resnet.pt.tar \
    --batch_normalization basic \
    --feature_adaptation film \
    --way 10 \
    --shot 2 \
    --query_test 5 \
    --l2_regularize_classifier \
    --top_k 1 \
    --selection_mode drop \
    --importance_mode loo \
    --kernel_agg class  \
    --tasks 10 \
    --spread_constraint none \
    --task_type generate_coreset_discard


#    --dataset cifar10 \
#    --data_path /scratches/stroustrup/jfb54/tf-meta-dataset/records \
