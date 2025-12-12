
accelerate launch --num_processes=2 --main_process_port 29500 modernvbert/src/modernvbert/contrastive_training/train.py \
     -c modernvbert/src/modernvbert/contrastive_training/modernvbert/doc/config/colmodel_hardnegs_text_300k_no_vdr_from_base.yaml > log/colmodel_hardnegs_text_300k_no_vdr_from_base.log 2>&1 &



# accelerate launch --num_processes=2 --main_process_port 29501 modernvbert/src/modernvbert/contrastive_training/train.py \
#      -c modernvbert/src/modernvbert/contrastive_training/modernvbert/doc/config/colmodel_hardnegs_text_300k_no_vdr_from_base_sparse.yaml > log/colmodel_hardnegs_text_300k_no_vdr_from_base_sparse.log 2>&1 &


accelerate launch --num_processes=2 modernvbert/src/modernvbert/contrastive_training/train.py \
     -c colpali/scripts/configs/modernvbert/train_colmodernvbert_m2.yaml
# pip install dacite