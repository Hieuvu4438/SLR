CUDA_VISIBLE_DEVICES=0 \
torchrun --nproc_per_node=1 \
--master_port 6665 \
main_task_retrieval.py \
--do_train \
--data_path data_ph \
--datatype ph \
--features_path /home/haipd/SLR/artifacts/sign_features/ph_domain_agnostic \
--features_path_retrain /home/haipd/SLR/artifacts/sign_features/ph_domain_aware \
--output_dir output_clcl