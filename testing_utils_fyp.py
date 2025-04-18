import os
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os
from datetime import datetime
from metrics import pt_psnr, calculate_ssim, calculate_psnr
from pytorch_msssim import ssim
from utils import save_rgb
today = datetime.today().strftime("%d_%m_%Y")
import random

DEST_PATH = "/eval_results/32_channels_performance_correct.txt"

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

    print(f"Saving performance statistics to {dest_path}")
    with open(dest_path, "a") as f:
        f.write(text)
        f.write("\n")
    print("Saving performance statistics done!")

def return_map(deg):
    if "rain" in deg:
        return "deraining"
    elif "noise" in deg or "nois" in deg:
        return "denoising"
    elif "haze" in deg:
        return "dehazing"
    else:
        return "deraining"

def get_wrong_degradation(prompt):
    degradations = ["noise", "rain", "haze"]
    degradations.remove(prompt)
    return random.choice(degradations)

def augment_prompt(prompt):
    ### special prompts
    lol_prompts = ["fix the illumination", "increase the exposure of the photo", "the image is too dark to see anything, correct the photo", "poor illumination, improve the shot", "brighten dark regions", "make it HDR", "improve the light of the image", "Can you make the image brighter?"]
    sr_prompts  = ["I need to enhance the size and quality of this image.", "My photo is lacking size and clarity; can you improve it?", "I'd appreciate it if you could upscale this photo.", "My picture is too little, enlarge it.", "upsample this image", "increase the resolution of this photo", "increase the number of pixels", "upsample this photo", "Add details to this image", "improve the quality of this photo"]
    en_prompts  = ["make my image look like DSLR", "improve the colors of my image", "improve the contrast of this photo", "apply tonemapping", "enhance the colors of the image", "retouch the photo like a photograper"]

    init = np.random.choice(["Remove the", "Reduce the", "Clean the", "Fix the", "Remove", "Improve the", "Correct the",])
    end  = np.random.choice(["please", "fast", "now", "in the photo", "in the picture", "in the image", ""])
    newp = f"{init} {prompt} {end}"

    if "lol" in prompt:
        newp = np.random.choice(lol_prompts)
    elif "sr" in prompt:
        newp = np.random.choice(sr_prompts)
    elif "en" in prompt:
        newp = np.random.choice(en_prompts)

    newp = newp.strip().replace("  ", " ").replace("\n", "")
    return newp

def test_model(model, language_model, lm_head, testsets, device, promptify, savepath="results/", initial_message=""):

    model.eval()
    if language_model:
        language_model.eval()
        lm_head.eval()

    DEG_ACC = []
    derain_datasets = ['Rain100L', 'Rain100H', 'Test100', 'Test1200', 'Test2800']

    statistics_message = initial_message

    with torch.no_grad():

        for testset in testsets:

            if savepath:
                dt_results_path = os.path.join(savepath, testset.name)
                if not os.path.exists(dt_results_path):
                    os.makedirs(dt_results_path, exist_ok=True)

            eval_message = f"\n>>> Eval on {testset.name} for {testset.degradation}(class={testset.deg_class})\n"
            statistics_message += eval_message

            print(eval_message)

            testset_name = testset.name
            test_dataloader = DataLoader(testset, batch_size=1, num_workers=4, drop_last=True, shuffle=False)
            psnr_dataset = []
            ssim_dataset = []
            psnr_noisy   = []
            use_y_channel= False

            if testset.name in derain_datasets:
                use_y_channel = True
                psnr_y_dataset = []
                ssim_y_dataset = []

            statistics_message += "The input human instructions (first 5):\n"

            for idx, batch in enumerate(test_dataloader):

                x = batch[0].to(device) # HQ image
                y = batch[1].to(device) # LQ image
                f = batch[2][0]         # filename
                # print(f"")
                t = [promptify(return_map(testset.degradation)) for _ in range(x.shape[0])]
                # t = t.to(device)
                # statistics_message += "The input human instructions (first 5):\n"
                if language_model:
                    # statistics_message += "The input human instructions (first 5):\n"
                    if idx < 5:
                        # print the input prompt for debugging
                        statistics_message += f"{t}\n"
                        print("\nInput prompt:", t)


                    lm_embd = language_model(t)
                    lm_embd = lm_embd.to(device)
                    text_embd, deg_pred = lm_head(lm_embd)
                    # text_embd = text_embd.to(device)
                    x_hat = model(y, text_embd)

                psnr_restore = torch.mean(pt_psnr(x, x_hat))
                psnr_dataset.append(psnr_restore.item())
                ssim_restore = ssim(x, x_hat, data_range=1., size_average=True)
                ssim_dataset.append(ssim_restore.item())
                psnr_base    = torch.mean(pt_psnr(x, y))
                psnr_noisy.append(psnr_base.item())

                if use_y_channel:
                    _x_hat = np.clip(x_hat[0].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)
                    _x     = np.clip(x[0].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)
                    _x_hat = (_x_hat*255).astype(np.uint8)
                    _x     = (_x*255).astype(np.uint8)

                    psnr_y = calculate_psnr(_x, _x_hat, crop_border=0, input_order='HWC', test_y_channel=True)
                    ssim_y = calculate_ssim(_x, _x_hat, crop_border=0, input_order='HWC', test_y_channel=True)
                    psnr_y_dataset.append(psnr_y)
                    ssim_y_dataset.append(ssim_y)

                ## SAVE RESULTS
                if savepath:
                    restored_img = np.clip(x_hat[0].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)
                    img_name = f.split("/")[-1]
                    save_rgb(restored_img, os.path.join(dt_results_path, img_name))

            if len(psnr_dataset) > 0:
                result_str = f"{testset_name}_base {np.mean(psnr_noisy)} Total images: {len(psnr_dataset)}\n{testset_name}_psnr {np.mean(psnr_dataset)}\n{testset_name}_ssim {np.mean(ssim_dataset)}\n"
                print(result_str)
                y_channel_result_str = ""
                if use_y_channel:
                    y_channel_result_str =f"{testset_name}_psnr-Y {np.mean(psnr_y_dataset)} {len(psnr_y_dataset)}\n{testset_name}_ssim-Y {np.mean(ssim_y_dataset)}\n"
                    print(y_channel_result_str)

                statistics_message += result_str
                statistics_message += y_channel_result_str
                divide_line = 25 * "***"
                statistics_message += divide_line
                statistics_message += "\n\n"
                print(); print(divide_line)

                del test_dataloader,psnr_dataset, psnr_noisy; gc.collect()
    save_performance_to_file(DEST_PATH, statistics_message)