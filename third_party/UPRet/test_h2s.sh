CUDA_VISIBLE_DEVICES=0 \
torchrun --nproc_per_node=1 \
--master_port 6669 \
main_task_retrieval.py \
--do_eval \
--datatype h2s \
--data_path data_h2 \
--features_path /home/haipd/SLR/artifacts/sign_features/h2s_domain_agnostic \
--features_path_retrain /home/haipd/SLR/artifacts/sign_features/h2s_domain_aware \
--output_dir output_h2s_eval \
${1:+--init_model "$1"}