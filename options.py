import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--patch_size', type=int, default=256, help='patchsize of input.')
parser.add_argument('--instructir_batch_size', type=int, default=32, help='batch size of InstructIR.')
parser.add_argument('--dataloader_batch_size', type=int, default=32, help='batch size of dataloader.')


parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for training')

parser.add_argument('--num_workers', type=int, default=8, help='number of workers.')
parser.add_argument("--checkpoint_dir",type=str, default="train_ckpt/",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument('--lm_head',      type=str, default="models/lm_instructir-7d.pt", help='Path to the language model weights')
parser.add_argument('--config',  type=str, default='configs/eval5d.yml', help='Path to config file')
parser.add_argument('--device',  type=int, default=0, help="GPU device")

parser.add_argument("--wblogger",type=str,default="instructir",help = "Determine to log to wandb or not and the project name")

parser.add_argument('--data_file_dir',  type=str, default="train_data_names/", help="Files that contains the training file names")
parser.add_argument('--denoise_dir',  type=str, default="data/Train/denoise/", help="Directory containing the images for denoise")
parser.add_argument('--dehaze_dir',  type=str, default="data/Train/dehaze/", help="Directory containing the images for dehaze")
parser.add_argument('--derain_dir',  type=str, default="data/Train/derain/", help="Directory containing the images for derain")
parser.add_argument('--de_type', nargs='+', default=['derain', 'denoise_15', 'dehaze', 'denoise_25', 'denoise_50'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--trained_model_weights', type=str, default="trained_weights/", help="File name of model state_dict()")
parser.add_argument('--image_model', type=str, default="models/im_instructir-7d.pt", help='Path to the language model weights')
parser.add_argument('--initial_lr',  type=float, default=5e-4, help="Learning Rate for optimizer")
parser.add_argument('--warmup_lr',  type=float, default=5e-6, help="Learning Rate for Scheduler")
parser.add_argument('--eta_min',  type=float, default=5e-6, help="Final Learning Rate for Scheduler")
parser.add_argument('--warmup_epochs',  type=int, default=20, help="warmup epochs for scheduler")
parser.add_argument('--chkpt_epoch',  type=int, default=20, help="Epochs for each Checkpoint ")
