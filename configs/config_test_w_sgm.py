
testset_root = '/home/jiaming/atd12k_points/test_2k_540p'
test_flow_root = '/home/jiaming/sgm/test_2k_540p'
test_annotation_root = 'datasets/test_2k_annotations'

trainset_root = '/home/jiaming/atd12k_points/train_10k'
train_flow_root = '/home/jiaming/sgm/train_10k'

test_size = (512, 288)
test_crop_size = (512, 288)
train_size = (512, 288)
train_crop_size = (512, 288)

mean = [0., 0., 0.]
std  = [1, 1, 1]

inter_frames = 1

model = 'AnimeInterp'
pwc_path = None

checkpoint = 'checkpoints/anime_interp_full.ckpt'

loss = '1*L1'
lr = 2e-4
max_epoch = 20

cuda = True
random_seed = 103

store_path = 'outputs/avi_full_results'



