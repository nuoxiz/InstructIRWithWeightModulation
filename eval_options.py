import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--patch_size', type=int, default=256, help='patchsize of input.')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of input.')

parser.add_argument('--num_workers', type=int, default=8, help='number of workers.')
parser.add_argument("--checkpoint_dir",type=str, default="train_ckpt/",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument('--lm',      type=str, default="models/lm_instructir-7d.pt", help='Path to the language model weights')
parser.add_argument('--config',  type=str, default='configs/eval5d.yml', help='Path to config file')

parser.add_argument('--promptify', type=str, default="simple_augment")
parser.add_argument('--debug',   action='store_true', help="Debug mode")
parser.add_argument('--save',    type=str, default='performance_results/64channels/modified_correct_instructions', help="Path to save the resultant images")

eval_args = parser.parse_args([])