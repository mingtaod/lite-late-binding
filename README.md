# lite-late-binding

## Commands to run for running evaluation
Ontonotes: 
CUDA_VISIBLE_DEVICES=X nohup python3 -u eval.py --path XXX --test ontonotes_data/processed/g_dev_tree_processed.json --batch 4 --check roberta-large-mnli &> XXX.out &

Figer: *
CUDA_VISIBLE_DEVICES=X nohup python3 -u eval.py --path XXX --test figer_data/processed/dev_processed.json --batch 4 --check roberta-large-mnli &> XXX.out &

Figon: *
CUDA_VISIBLE_DEVICES=X nohup python3 -u eval.py --path XXX --test figer_ontonotes_data/dev.json --batch 4 --check roberta-large-mnli &> XXX.out &
