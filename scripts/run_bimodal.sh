ulimit -n 50000
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=3

python3 ./learners/protonets_attention/src/main.py \
    --checkpoint_dir /scratches/stroustrup/etv21/debug \
    --classifier protonets_attention \
    --mode test \
    --test_datasets cifar100 \
    --data_path /scratches/stroustrup/jfb54/tf-meta-dataset/records \
    --test_model_path /scratches/stroustrup/etv21/protonets_attention/checkpoints_l2/fully_trained.pt \
    --pretrained_resnet_path learners/protonets_attention/models/pretrained_resnet.pt.tar \
    --batch_normalization basic \
    --feature_adaptation film \
    --way 6 \
    --shot 5 \
    --query_test 10 \
    --dataset cifar100 \
    --l2_regularize_classifier \
    --top_k 6 \
    --selection_mode top_k \
    --importance_mode all \
    --kernel_agg class  \
    --tasks 2 \
    --spread_constraint none \
    --test_case bimodal \
    --task_type shot_selection
    
