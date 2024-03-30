ulimit -n 50000
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1


python3 ./learners/protonets_attention/src/main.py \
    --classifier protonets_attention \
    --mode test \
    --data_path /scratches/bezier_2/jfb54/tf-meta-dataset/records \
    --dataset cifar10 \
    --checkpoint /scratch/etv21/large_scale_loo/debug \
    --test_model_path /scratches/stroustrup/etv21/protonets_attention/checkpoints_l2/fully_trained.pt \
    --pretrained_resnet_path learners/protonets_attention/models/pretrained_resnet.pt.tar \
    --batch_normalization basic \
    --feature_adaptation film \
    --shot 800 \
    --way 2 \
    --query_test 199 \
    --l2_regularize_classifier \
    --drop_rate 0.2 \
    --selection_mode drop \
    --importance_mode loo \
    --kernel_agg class  \
    --tasks 1 \
    --spread_constraint none \
    --task_type noisy_shots \
    --error_rate 0.2 \
    --noise_type mislabel \


#    --dataset cifar10 \
#    --data_path /scratches/stroustrup/jfb54/tf-meta-dataset/records \
