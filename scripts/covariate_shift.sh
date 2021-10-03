ulimit -n 50000
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0


python3 ./learners/protonets_attention/src/main.py \
    --classifier protonets_attention \
    --mode test \
    --data_path /scratches/stroustrup/jfb54/tf-meta-dataset/records \
    --dataset cifar10 \
    --checkpoint /scratches/stroustrup/etv21/debug \
    --test_model_path /scratches/stroustrup/etv21/protonets_attention/checkpoints_l2/fully_trained.pt \
    --pretrained_resnet_path learners/protonets_attention/models/pretrained_resnet.pt.tar \
    --batch_normalization basic \
    --feature_adaptation film \
    --shot 4 \
    --way 8 \
    --query_test 10 \
    --l2_regularize_classifier \
    --drop_rate 0.1 \
    --selection_mode drop \
    --importance_mode all \
    --kernel_agg class  \
    --tasks 1 \
    --spread_constraint none \
    --task_type noisy_shots \
    --error_rate 0.1 \
    --noise_type ood \


#    --dataset cifar10 \
#    --data_path /scratches/stroustrup/jfb54/tf-meta-dataset/records \
