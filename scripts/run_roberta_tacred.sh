for SEED in 78 23 61;
do python train_tacred.py --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta;
done;