import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
import random
import gc
from datetime import datetime

from utils import *
from models.instructir import create_model

from text.models import LanguageModel, LMHead

from eval_options import eval_args
from testing_utils_fyp import test_model

def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


if __name__=="__main__":

    SEED=42
    seed_everything(SEED=SEED)
    torch.backends.cudnn.deterministic = True

    now = datetime.now()

    # GPU        = eval_args.device
    DEBUG      = eval_args.debug
    IMAGE_MODEL_NAME = "models/instructir_with_weight_modulation_32ch.pt"

    CONFIG     = eval_args.config
    LM_HEAD_MODEL   = eval_args.lm
    SAVE_PATH  = eval_args.save


    testing_message = f"{now.strftime('%Y-%m-%d %H:%M:%S')} Testing Re-trained InstructIR Model 32 Channels Correct Human Instructions \n"

    print ('CUDA GPU available: ', torch.cuda.is_available())

    device = torch.device(f'cuda' if torch.cuda.is_available() else "cpu")
    print(f"\n\n*************** device: {device} ***************\n\n")
    print('CUDA visible devices: ' + str(torch.cuda.device_count()))
    if torch.cuda.is_available():
        print('CUDA current device: ', torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))

    # parse config file
    with open(os.path.join(CONFIG), "r") as f:
        config = yaml.safe_load(f)
    cfg = dict2namespace(config)

    use_offset = True

    testing_message += f"---- Weight Modification Enabled: {use_offset}\n"

    print(f"---- Weight Modification Enabled: {use_offset}\n")

    infor = f"Image Model Name: {IMAGE_MODEL_NAME}, Projection Head: {LM_HEAD_MODEL}\nDevice: {device}, Human Instruction Source: {eval_args.promptify}\nConfig: {CONFIG}\n\n"

    testing_message += infor

    print(20*"****")
    print("EVALUATION")
    print(infor)
    print(20*"****")

    ################### TESTING DATASET

    TESTSETS = []
    denoise_testsets   = []
    rain_testsets = []
    haze_testsets = []
    # Denoising
    printed_noise_path = True
    try:
        for testset in cfg.test.dn_datasets:
            for sigma in cfg.test.dn_sigmas:
                noisy_testpath = os.path.join(cfg.test.dn_datapath, testset+ f"_{sigma}")
                clean_testpath = os.path.join(cfg.test.dn_datapath, testset)
                if printed_noise_path:
                    print(f"clean_testpath:{clean_testpath}, noisy_testpath:{noisy_testpath} ")
                    printed_noise_path = False
                denoise_testsets.append([clean_testpath, noisy_testpath])
    except:
        denoise_testsets = []

    printed_rain_path = True
    # RAIN
    try:
        for noisy_testpath, clean_testpath in zip(cfg.test.rain_inputs, cfg.test.rain_targets):
            if printed_rain_path:
                print(f"clean_testpath:{clean_testpath}, noisy_testpath:{noisy_testpath} ")
                printed_rain_path = False
            rain_testsets.append([clean_testpath, noisy_testpath])
    except:
        rain_testsets = []

    # HAZE
    try:
        haze_testsets = [[cfg.test.haze_targets, cfg.test.haze_inputs]]
    except:
        haze_testsets = []

    TESTSETS += denoise_testsets
    TESTSETS += rain_testsets
    TESTSETS += haze_testsets

    if len(denoise_testsets) > 0:
        testing_message += f"Denoise testset length: {len(denoise_testsets)}\n"

    if len(rain_testsets) > 0:
        testing_message += f"Derain testset length: {len(rain_testsets)}\n"

    if len(haze_testsets) > 0:
        testing_message += f"Dehaze testset length: {len(haze_testsets)}\n"

    testset_len = f"Total testsets length: {len(TESTSETS)}\n"

    testing_message += testset_len

    print (testset_len)
    print (20 * "----")


    ################### RESTORATION MODEL

    print ("Creating InstructIR")
    model = create_model(input_channels =cfg.model.in_ch, width=cfg.model.width, enc_blks = cfg.model.enc_blks,
                    middle_blk_num = cfg.model.middle_blk_num, dec_blks = cfg.model.dec_blks,
                    txtdim=cfg.model.textdim, include_offset=use_offset)

    ################### LOAD IMAGE MODEL

    assert IMAGE_MODEL_NAME, "Model weights required for evaluation"

    print ("IMAGE MODEL CKPT:", IMAGE_MODEL_NAME)
    model.load_state_dict(torch.load(IMAGE_MODEL_NAME, map_location=device), strict=True)
    for param in model.parameters():
        param.requires_grad = True
    model = model.to(device)
    total_params = count_params(model)

    # Freeze prompt generator blocks
    for prompt_block in model.promptBlocks:
        for param in prompt_block.parameters():
            param.requires_grad = False

    # Freeze offset generators
    # (Assuming self.offset_generators is a list of OffsetGenerator modules)
    for offset_gen in model.offset_generators:
        for param in offset_gen.parameters():
            param.requires_grad = False


    for param in model.prompt_block_middle_blks.parameters():
        param.requires_grad = False

    for param in model.middleblock_offsetGen.parameters():
        param.requires_grad = False

    base_ir_model_params = count_params(model)

    for param in model.parameters():
      param.requires_grad = False

    # enable prompt generator blocks
    for prompt_block in model.promptBlocks:
        for param in prompt_block.parameters():
            param.requires_grad = True

    # enable offset generators
    for offset_gen in model.offset_generators:
        for param in offset_gen.parameters():
            param.requires_grad = True

    for param in model.prompt_block_middle_blks.parameters():
        param.requires_grad = True

    for param in model.middleblock_offsetGen.parameters():
        param.requires_grad = True


    params_of_modification = count_params(model)

    for param in model.parameters():
      param.requires_grad = True


    model_params_message = f"""\nTotal params: {total_params/1e6}M,
    base image restoration model params: {base_ir_model_params/1e6}M\n
    OffsetGenerator & PromptGenBlock params: {params_of_modification/1e6}M\n"""

    testing_message += model_params_message

    # nparams = count_params(model)
    print(model_params_message)
    ################### LANGUAGE MODEL

    # try:
    #     PROMPT_DB  = cfg.llm.text_db
    # except:
    #     PROMPT_DB  = None

    if cfg.model.use_text:
        # os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Initialize the LanguageModel class
        LMODEL = cfg.llm.model
        language_model = LanguageModel(model=LMODEL).to(device)
        lm_head = LMHead(embedding_dim=cfg.llm.model_dim, hidden_dim=cfg.llm.embd_dim, num_classes=cfg.llm.nclasses)
        lm_head = lm_head.to(device)
        # language_model = language_model.to(device)
        lm_nparams   = count_params(lm_head)

        print("LMHEAD MODEL CKPT:", LM_HEAD_MODEL)
        lm_head.load_state_dict(torch.load(LM_HEAD_MODEL, map_location=device), strict=True)
        print("Loaded weights!")

    else:
        LMODEL = None
        language_model = None
        lm_head = None
        lm_nparams = 0

    print (20 * "----")

    ################### TESTING !!

    from datasets import augment_prompt, create_testsets

    if eval_args.promptify == "simple_augment":
        promptify = augment_prompt
    elif eval_args.promptify == "chatgpt":
        instruction_file_path = "/content/drive/MyDrive/FYPData/human_instructions.json"
        with open(instruction_file_path, "r") as f:
            prompts = json.load(f)
        for deg in prompts.keys():
            random.shuffle(prompts[deg])

        def promptify(deg):
            return random.choice(prompts[deg])

        print("--- Using ChatGPT generated human instructions\n")
        testing_message += "--- Using ChatGPT generated human instructions\n"
    else:
        def promptify(deg):
            return eval_args.promptify


    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

    test_datasets = create_testsets(TESTSETS, debug=True)

    test_model(model, language_model, lm_head, test_datasets, device, promptify, savepath=eval_args.save, initial_message=testing_message)
