ulimit -n 50000
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1

for i in 25
do
    mkdir /scratches/stroustrup/etv21/protonets_attention/no_restrict/track_weights
    python3 ./learners/protonets_attention/src/main.py \
        --checkpoint_dir /scratches/stroustrup/etv21/protonets_attention/no_restrict/track_weights \
        --classifier protonets_attention \
        --mode test \
        --test_datasets ilsvrc_2012 \
        --data_path /scratches/stroustrup/jfb54/tf-meta-dataset/records \
        --test_model_path /scratches/stroustrup/etv21/protonets_attention/checkpoints_l2/fully_trained.pt \
        --pretrained_resnet_path learners/protonets_attention/models/pretrained_resnet.pt.tar \
        --batch_normalization basic \
        --feature_adaptation film \
        --max_support_test 10 \
        --max_way_test 5 \
        --way 5 \
        --shot 10 \
        --query_test 10 \
        --dataset ilsvrc_2012 \
        --l2_regularize_classifier \
        --top_k $i \
        --selection_mode divine \
        --importance_mode all \
        --kernel_agg class  \
        --tasks 1 \
        --spread_constraint none
done