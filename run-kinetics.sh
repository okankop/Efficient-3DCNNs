 python main.py --root_path ~/ \
 	--video_path ~/datasets/Kinetics \
 	--annotation_path Efficient-3DCNNs/annotation_Kinetics/kinetics.json \
 	--result_path Efficient-3DCNNs/results \
 	--resume_path Efficient-3DCNNs/results/kinetics_mobilenetv2_0.45x_RGB_16_best.pth \
 	--dataset kinetics \
 	--sample_size 112 \
 	--n_classes 600 \
 	--model mobilenetv2 \
 	--version 1.1 \
 	--groups 3 \
 	--width_mult 0.45 \
 	--train_crop random \
 	--learning_rate 0.1 \
 	--sample_duration 16 \
 	--batch_size 64 \
 	--n_threads 16 \
 	--checkpoint 1 \
 	--n_val_samples 1 \
 	# --no_train \
 	# --no_val \
 	# --test