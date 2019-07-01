
python main.py --root_path ~/ \
	--video_path ~/datasets/jester \
	--annotation_path Efficient-3DCNNs/annotation_Jester/jester.json \
	--result_path Efficient-3DCNNs/results \
	--dataset jester \
	--n_classes 27 \
	--model mobilenet \
	--groups 3 \
	--width_mult 0.5 \
	--train_crop random \
	--learning_rate 0.1 \
	--sample_duration 16 \
	--downsample 2 \
	--batch_size 16 \
	--n_threads 16 \
	--checkpoint 1 \
	--n_val_samples 1 \
	# --no_train \
 	# --no_val \
 	# --test