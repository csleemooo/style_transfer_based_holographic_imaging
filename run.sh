## polystyrene_bead experiments ##
python train_holography_style_transfer.py --data_name polystyrene_bead --save_dir ./experiments/polystyrene_bead/multi_style --log_dir ./logs/polystyrene_bead/multi_style --device cuda:1 --batch_size 8
python train_holography_style_transfer.py --data_name MNIST --save_dir ./experiments --log_dir ./logs --exp_name half_style_half_content --device cuda:1 --batch_size 8

## test code
python test_interpolation.py --data_name MNIST --exp_name half_style_half_content --device cuda:1 --test_interpolation 1 # see interpolation of AdaIN
python test_interpolation.py --data_name MNIST --exp_name half_style_half_content --device cuda:1 --test_interpolation 0

# experimental dataset: polystyrene_bead field retrieval
python train_holography_field_retrieval_disc.py --data_name polystyrene_bead --save_dir ./experiments --log_dir ./logs --exp_name half_style_half_content_disc --holo_weight 10 --device cuda:0 --content_weight 0.0 --style_weight 1.0 --identity_weight 10 --unknown_distance 1 --batch_
size 4
python train_holography_field_retrieval_disc.py --data_name polystyrene_bead --save_dir ./experiments --log_dir ./logs --exp_name single_style_disc --holo_weight 10 --device cuda:1 --content_weight 0.0 --style_weight 1.0 --identity_weight 10 --unknown_distance 1 --batch_size 4

# simulation dataset: MNIST field retrieval
python train_holography_field_retrieval_disc.py --data_name MNIST --save_dir ./experiments --log_dir ./logs --exp_name single_style_disc --holo_weight 10 --device cuda:1 --content_weight 0.0 --style_weight 1.0 --identity_weight 10 --unknown_distance 1 --batch_size 8
python train_holography_field_retrieval_disc.py --data_name MNIST --save_dir ./experiments --log_dir ./logs --exp_name half_style_half_content_disc --holo_weight 10 --device cuda:1 --content_weight 0.0 --style_weight 1.0 --identity_weight 10 --unknown_distance 1 --batch_size 8
python train_holography_field_retrieval_disc.py --data_name MNIST --save_dir ./experiments --log_dir ./logs --exp_name half_style_half_content_disc2 --holo_weight 10 --device cuda:1 --content_weight 0.0 --style_weight 10.0 --identity_weight 10 --unknown_distance 1 --batch_size 8


# current best simulation result
# 241105
python train_holography_field_retrieval_disc.py --data_name MNIST --save_dir ./experiments --log_dir ./logs --exp_name 241104_half_style_half_content_disc --holo_weight 10 --device cuda:1 --content_weight 0.0 --style_weight 10.0 --identity_weight 10 --unknown_distance 1 --batch_size 32
tensorboard --logdir="/mnt/mooo/CS/style transfer based holographic imaging/code/pytorch-AdaIN/logs/MNIST/241104_half_style_half_content_disc_field_retrieval" --port=9506
python test_field_retrieval.py --device cpu