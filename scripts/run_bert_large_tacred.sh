for SEED in 78 23 61;
do python train_tacred.py --model_name_or_path bert-large-cased --input_format typed_entity_marker --seed $SEED --run_name bert-large;
done;