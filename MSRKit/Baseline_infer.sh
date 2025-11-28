# Special for percussions
python ../AoMSS/inferenceForPerc.py --input_folder ../AoMSS/BSR_output/Percussions --store_dir ../CUPAudioGroup/Percussions --config_path ../pretrain/Drumsep/config_mdx23c.yaml --start_check_point ../pretrain/Drumsep/drumsep_5stems_mdx23c_jarredou.ckpt
# All the other instruments
python inference.py