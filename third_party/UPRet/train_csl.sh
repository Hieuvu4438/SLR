CUDA_VISIBLE_DEVICES=0 \
torchrun --nproc_per_node=1 \
--master_port 6666 \
main_task_retrieval.py \
--do_train \
--data_path data_csl \
--alpha 0.8 \
--datatype csl \
--features_path /home/haipd/SLR/artifacts/sign_features/csl_domain_agnostic \
--features_path_retrain /home/haipd/SLR/artifacts/sign_features/csl_domain_aware \
--output_dir output_csl