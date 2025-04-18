import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
import yaml
import random
import json
from datetime import datetime

from training_utils.TrainDataset import InstructIRTrainDataset
from training_utils.scheduler import LinearWarmupCosineAnnealingLR
import sys
from options import parser
from models.instructir import InstructIR, create_model
from text.models import LanguageModel, LMHead


def add_date_to_filename(filepath):
    # Get today's date formatted as day_month_year (e.g. 08_03_2025)
    today = datetime.today().strftime("%d_%m_%Y")
    # Split the filepath into the base and extension parts
    base, ext = os.path.splitext(filepath)
    # Construct the new filepath by inserting the date before the extension
    new_filepath = f"{base}_{today}{ext}"
    return new_filepath

def create_training_checkpoint_folder(base_folder):
    """
    Given a base folder path, this function creates a directory structure:
    base_folder/train_date_DD_MM_YYYY/train_time_HH_MM
    using today's date and the current time, then returns the new folder path.

    Args:
        base_folder (str): The path to the base checkpoint folder.

    Returns:
        str: The full path of the created training checkpoint folder.
    """
    # Format today's date as DD_MM_YYYY (e.g., "08_03_2025")
    today_date = datetime.today().strftime("%d_%m_%Y")
    # Format current time as HH_MM (e.g., "01_20")
    start_time = datetime.now().strftime("%H_%M")

    # Construct the new folder path
    new_folder_path = f"{base_folder}/train_date_{today_date}/train_time_{start_time}"

    return new_folder_path

DEST_PATH = "train_loss/train_loss.txt"

print(f"DEST_PATH: {DEST_PATH}")

def save_performance_to_file(dest_path, text):
    """
    Append the given text to the file at dest_path.
    If the file does not exist, it is created.
    If the folder does not exist, it is created.

    Args:
        dest_path (str): The path to the destination file.
        text (str): The text to be appended to the file.
    """
    # Extract the directory from the destination path.
    directory = os.path.dirname(dest_path)
    # if directory and not os.path.exists(directory):
    os.makedirs(directory, exist_ok=True)

    print(f"Saving Training Loss to {dest_path}")
    with open(dest_path, "a") as f:
        f.write(text)
        f.write("\n")
    print("Saving Training Loss done!")


def train_instructir_model(instructir_model, language_model, lm_head, train_data_loader,human_instructions, optimizer, scheduler, loss_fn, opt, device,
    accumulation_steps, image_batch_size, ini_message = "Stage One:", check_point_epoch=20):
    """
    Train the InstructIR model.
    Args:
        instructir_model: The image restoration model to train.
        language_model: A language model that has a tokenizer and a model (e.g., Hugging Face model).
        lm_head: A head applied on the language model output (e.g., to get text embeddings).
        train_data_loader: DataLoader yielding training batches.
        human_instructions: List of human instruction strings.
        optimizer: The optimizer (e.g., Adam).
        scheduler: Learning rate scheduler.
        loss_fn: Loss function (e.g., L1 loss).
        opt: An object with training options (must contain opt.epochs and opt.checkpoint_dir).
        device: The torch device (e.g., 'cuda' or 'cpu').
        accumulation_steps: Number of batches over which to accumulate gradients.
        image_batch_size: The number of images per batch (used to generate instructions).
    Returns:
        The trained instructir_model.
    """
    base = 0
    save_performance_to_file(DEST_PATH, ini_message)
    checkpoint_dir = create_training_checkpoint_folder(opt.checkpoint_dir)
    for epoch in range(opt.epochs):
        now = datetime.now()
        epoch_msg = f"{now.strftime('%Y-%m-%d %H:%M:%S')} --- Start of Epoch: {epoch + 1 + base}\n"
        print(epoch_msg)

        instructir_model.train()
        running_loss = 0.0
        # accumulated_loss = 0.0  # Initialize accumulated loss
        for batch_idx, batch in enumerate(train_data_loader):
            # Unpack batch
            [clean_name, deg_id], degrad_patch, clean_patch = batch
            # print(f"clean name: {clean_name}")
            # Get a random human instruction for each image in the batch
            human_instruction = [random.choice(human_instructions) for _ in range(image_batch_size)]

            lm_embd = language_model(human_instruction)
            lm_embd = lm_embd.to(device)
            text_embd, deg_pred = lm_head(lm_embd)
            # Forward pass through the InstructIR model
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)
            restored_image = instructir_model(degrad_patch, text_embd)
            # Compute loss
            loss = loss_fn(restored_image, clean_patch)

            optimizer.zero_grad()
            # Backpropagate
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if( batch_idx + 1) % 100 == 0:
                batch_msg = f"Epoch [{epoch + 1 + base}/{opt.epochs}], Batch [{batch_idx+1}/{len(train_data_loader)}], Running Avg Loss: {running_loss/(batch_idx+1):.20f}\n"
                # print(batch_msg)
                if( batch_idx + 1) % 250 == 0:
                    print(batch_msg)
                epoch_msg += batch_msg

        # Step the scheduler after each epoch
        if scheduler is not None:
            scheduler.step(epoch)
        avg_epoch_loss = running_loss / len(train_data_loader)
        avg_epoch_loss_str = f"Epoch [{epoch + 1 + base}/{opt.epochs }], Average Loss: {avg_epoch_loss:.20f}\n"
        epoch_msg += avg_epoch_loss_str
        print(avg_epoch_loss_str)

        # Save a checkpoint for the current epoch
        if (epoch + 1 + base) % check_point_epoch == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path_model = os.path.join(checkpoint_dir, f"epoch_{epoch + 1 + base}_model.pth")
            checkpoint_path_scheduler = os.path.join(checkpoint_dir, f"epoch_{epoch + 1 + base}_scheduler.pth")
            checkpoint_path_optimizer = os.path.join(checkpoint_dir, f"epoch_{epoch + 1 + base}_optimizer.pth")

            torch.save(instructir_model.state_dict(), checkpoint_path_model)
            torch.save(scheduler.state_dict(), checkpoint_path_scheduler)
            torch.save(optimizer.state_dict(), checkpoint_path_optimizer)
            print(f"Checkpoint saved at {checkpoint_path_model}\n{checkpoint_path_scheduler}\n{checkpoint_path_optimizer}")
        # print end of epoch
        now = datetime.now()
        end_epoch_msg = f"{now.strftime('%Y-%m-%d %H:%M:%S')} --- End of Epoch: {epoch + 1 + base}\n"
        print(end_epoch_msg)
        epoch_msg += end_epoch_msg
        # save current epochs statistics
        save_performance_to_file(DEST_PATH, epoch_msg)
    return instructir_model

def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

def count_params(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad is True)
    return trainable_params

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

import os
def main():
    now = datetime.now()

    formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")

    training_message = f"{formatted_datetime} Training Log for Re-train InstructIR (output channel in OffsetGenertor conv = 32) - last 20 epochs\n\n"

    # torch.cuda.empty_cache()
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
    
    # Set seeds for reproducibility
    seed_everything()
    print(f"Torch version: {torch.__version__}")
    print(f"cuda version: {torch.version.cuda}")
    print("CUDA available:", torch.cuda.is_available())

    # Parse input arguments
    opt = parser.parse_args([])
    print("Options:")
    print(opt)


    LANGUAGE_HEAD = opt.lm_head
    CONFIG = opt.config
    IMAGE_BATCH_SIZE_DATALOADER = opt.dataloader_batch_size
    INITIAL_LEARNING_RATE = opt.initial_lr
    MAX_EPOCHS = opt.epochs
    WARMUP_EPOCHS = opt.warmup_epochs
    WARMUP_START_LR=opt.warmup_lr
    ETA_MIN = opt.eta_min
    CHECK_POINT_EPOCH = opt.chkpt_epoch
    accumulation_steps = opt.instructir_batch_size/IMAGE_BATCH_SIZE_DATALOADER

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n\n*************** device: {device} ***************\n\n")


    training_message += f"options: {opt}\n\n*************** device: {device} ***************\n\n"

    # Parse config file and convert to namespace for easy attribute access
    with open(os.path.join(CONFIG), "r") as f:
        config = yaml.safe_load(f)
    cfg = dict2namespace(config)

    # Load human instructions from JSON
    instruction_file_path = "text/human_instructions.json"
    with open(instruction_file_path, "r") as f:
        data = json.load(f)
    human_instructions = data["denoising"] + data["deraining"] + data["dehazing"]
    random.shuffle(human_instructions)
    print(f"Human Instruction Set Length: {len(human_instructions)}\n")

    training_message += f"Human Instruction Set Length: {len(human_instructions)}\n"

    # Create training dataset and dataloader
    training_dataset = InstructIRTrainDataset(args=opt)
    train_data_loader = DataLoader(
        training_dataset,
        batch_size=IMAGE_BATCH_SIZE_DATALOADER,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers,
        persistent_workers=True
    )


    training_message += f"Loading Language Model: {cfg.llm.model}"
    # Load language model and LM head
    print(f"\nLoading Language Model: {cfg.llm.model}")
    language_model = LanguageModel(model=cfg.llm.model).to(device)
    lm_head = LMHead(
        embedding_dim=cfg.llm.model_dim,
        hidden_dim=cfg.llm.embd_dim,
        num_classes=cfg.llm.nclasses
    ).to(device)

    lm_nparams = count_params(lm_head)
    print("Projection Head CKPT Path:", LANGUAGE_HEAD)
    lm_head.load_state_dict(torch.load(LANGUAGE_HEAD, map_location=device), strict=True)
    print("\nProjection Head loaded weights!", lm_nparams/1e6, "M\n")

    training_message += f"Projection Head CKPT Path: {LANGUAGE_HEAD}\nProjection Head loaded weights: {lm_nparams/1e6}M\nFreezing language model and project head"

    # Freeze parameters of the language model and LM head
    for param in language_model.parameters():
        param.requires_grad = False
    for param in lm_head.parameters():
        param.requires_grad = False

    print(f"------ Training for {opt.de_type}")

    training_message += f"------ Training for {opt.de_type}\nDataset:"

    noise_dataset_size = ""
    rain_dataset_size = ""
    haze_dataset_size = ""

    if ("denoise_15" in opt.de_type) or ("denoise_25" in opt.de_type) or ("denoise_50" in opt.de_type):
        noise_dataset_size = f"Noise sigma 15: {len(training_dataset.s15_ids)}, Noise sigma 25: {len(training_dataset.s25_ids)}, Noise sigma 50: {len(training_dataset.s50_ids)}\n"
    if "dehaze" in opt.de_type:
        haze_dataset_size = f"Haze Images Length: {len(training_dataset.hazy_ids)}\n"
    if "derain" in opt.de_type:
        rain_dataset_size = f"Rain Images Length: {len(training_dataset.rs_ids)}\n"

    training_message += noise_dataset_size
    training_message += haze_dataset_size
    training_message += rain_dataset_size




    print("----------------- Retrain InstructIR started... ----------------------")
    print("Start loading the pre-trained InstructIR model...")
    # Create the InstructIR model and move to device
    instructir_model = create_model(txtdim=256, middle_blk_num=4, width=opt.instructir_batch_size, include_offset=True)
    instructir_model = instructir_model.to(device)

    im_checkpoint_path = "models/im_instructir-7d.pt"

    instructir_stage_one_path = im_checkpoint_path
    print("\nInstructIR Path:", instructir_stage_one_path, "\n")
    instructir_model.load_state_dict(torch.load(instructir_stage_one_path, map_location=device), strict=False)

    training_message += f"\nInstructIR Path: {instructir_stage_one_path}\n"

    instructir_model_nparams = count_params(instructir_model)
    total_params_msg = f"\n---- InstructIR + Modifcation: {instructir_model_nparams/1e6}M parameters\n"

    for param in instructir_model.parameters():
        param.requires_grad = False

  # enable prompt generator blocks
    for prompt_block in instructir_model.promptBlocks:
        for param in prompt_block.parameters():
            param.requires_grad = True

  # enable offset generators
    for offset_gen in instructir_model.offset_generators:
        for param in offset_gen.parameters():
            param.requires_grad = True

    for param in instructir_model.prompt_block_middle_blks.parameters():
        param.requires_grad = True

    for param in instructir_model.middleblock_offsetGen.parameters():
        param.requires_grad = True


    params_to_train = list(instructir_model.promptBlocks.parameters()) + list(instructir_model.offset_generators.parameters()) + \
        list(instructir_model.prompt_block_middle_blks.parameters()) + list(instructir_model.middleblock_offsetGen.parameters())

    mod_params_msg = f"\n---- OffsetGenerators + PromptGenBlocks: {sum(p.numel() for p in params_to_train if p.requires_grad is True)/1e6}M parameters\n"

    print(total_params_msg, mod_params_msg)

    training_message += total_params_msg
    training_message += mod_params_msg

    assert count_params(instructir_model) == sum(p.numel() for p in params_to_train if p.requires_grad is True)

    scheduler = None
    # Set up new optimizer and learning rate scheduler
    optimizer = optim.AdamW(params_to_train,  lr=INITIAL_LEARNING_RATE)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,  warmup_epochs=WARMUP_EPOCHS, max_epochs=MAX_EPOCHS,
      warmup_start_lr=WARMUP_START_LR, eta_min=ETA_MIN )

    training_params = f"""\nLearning Rate: {INITIAL_LEARNING_RATE}, Image Model Batch_size: {opt.instructir_batch_size}, Dataloader Batch_size: {IMAGE_BATCH_SIZE_DATALOADER}, Acccumulation steps: {accumulation_steps}\n"""

    scheduler_setting = f"Scheduler: {str(scheduler)}\nmax_epochs: {MAX_EPOCHS}, warmup_start_lr:{WARMUP_START_LR}, eta_min: {ETA_MIN})"

    training_message += training_params
    training_message += scheduler_setting

    print(training_params, scheduler_setting)

    if scheduler is None:
        print("\nNot using scheduler")

    #################################################### Load optimizer, scheduler, instructIR model ##############################

    # optimizer_checkpoint_path = "/content/drive/MyDrive/FYPData/trained_weights/epoch_280_optimizer.pth"
    # scheduler_checkpoint_path = "/content/drive/MyDrive/FYPData/trained_weights/epoch_280_scheduler.pth"

    # # Load the optimizer checkpoint with map_location set to device
    # optimizer.load_state_dict(torch.load(optimizer_checkpoint_path, map_location=device))

    # # Load the scheduler checkpoint with map_location set to device
    # scheduler.load_state_dict(torch.load(scheduler_checkpoint_path, map_location=device))

    # training_message += f"Loaded optimizer checkpoint at: {optimizer_checkpoint_path}\nLoaded scheduler checkpoint at: {scheduler_checkpoint_path}"


    #################################################### Load optimizer, scheduler, instructIR model ##############################


    # Loss function
    loss_fn = nn.L1Loss()

    # train for the offset generator and promptblock
    instructir_model = train_instructir_model(instructir_model=instructir_model, language_model=language_model, lm_head=lm_head,
        train_data_loader=train_data_loader, human_instructions=human_instructions, optimizer=optimizer, scheduler=scheduler,
        loss_fn=loss_fn, opt=opt, device=device, accumulation_steps=accumulation_steps, image_batch_size=IMAGE_BATCH_SIZE_DATALOADER,
        ini_message=training_message, check_point_epoch=CHECK_POINT_EPOCH
    )

    # Save the final model weights
    model_weights_file_name = add_date_to_filename("instructir_weights_3d_300epochs_smaller_channel.pt")
    os.makedirs(opt.trained_model_weights, exist_ok=True)
    model_file_path = os.path.join(opt.trained_model_weights, model_weights_file_name)
    torch.save(instructir_model.state_dict(), model_file_path)
    print(f"Retraining finished, final model weights saved at: {model_file_path}\n")


if __name__ == "__main__":
    main()
