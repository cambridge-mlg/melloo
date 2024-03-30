ulimit -n 50000
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1

python3 ./learners/protonets_attention/src/main.py \
    --checkpoint_dir /scratch/etv21/debug \
    --classifier protonets_attention \
    --mode test \
    --test_datasets cifar10 \
    --data_path /scratches/stroustrup/etv21/records \
    --test_model_path /scratches/stroustrup/etv21/protonets_attention/checkpoints_l2/fully_trained.pt \
    --pretrained_resnet_path learners/protonets_attention/models/pretrained_resnet.pt.tar \
    --batch_normalization basic \
    --feature_adaptation film \
    --way 5 \
    --shot 5 \
    --query_test 10 \
    --dataset cifar10 \
    --l2_regularize_classifier \
    --top_k -1 \
    --selection_mode top_k \
    --importance_mode representer \
    --kernel_agg query_class  \
    --tasks 50 \
    --spread_constraint none \
    --task_type shot_selection \
    --indep_eval 
    
# revised_loo_exps/shot_selection/cifar10_5_way_5_shot/loo \

#    --test_case bimodal \
#    --checkpoint_dir /scratches/stroustrup/etv21/debug \
