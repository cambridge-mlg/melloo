ulimit -n 50000
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=5


python3 ./learners/protonets_attention/src/main.py \
    --classifier protonets_attention \
    --data_path /scratch/jfb54/tf-meta-dataset/records \
    --tasks_per_batch 16 \
    -c /scratch/etv21/protonets_attention/checkpoints_l2 \
    --batch_normalization basic \
    --feature_adaptation film \
    -i 30000 \
    --max_support_train 300 \
    --dataset meta-dataset_ilsvrc_only \
    --attention_temperature 2.0 \
    -lr 0.0005 \
    --pretrained_resnet_path learners/protonets_attention/models/pretrained_resnet.pt.tar \
    --l2_regularize_classifier       

