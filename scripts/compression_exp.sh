ulimit -n 50000
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

python3 ./learners/protonets/src/main.py \
	--data_path /scratches/stroustrup/jfb54/adv-fsl \
	--checkpoint_dir /scratch/etv21/debug \
	--dataset mini_imagenet \
	--mode test \
	--test_model_path ./learners/protonets/models/protonets_mini_imagenet_5-way_5-shot.pt  \
	--test_shot 100 --test_way 5 \
	--query 100
	--exp_type compress
	
	
#python3 ./learners/cnaps/src/run_cnaps.py \
#	--data_path /scratches/stroustrup/jfb54/tf-meta-dataset/records \
#	--checkpoint_dir /scratch/etv21/debug \
#	--feature_adaptation film \
#	--dataset ilsvrc_2012 \
#	--mode test \
#	-m learners/cnaps/models/meta-trained_meta-dataset_film.pt \
#	--shot 100 --way 5 \
#	--query_test 100