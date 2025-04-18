# from TrainDataset import InstructIRTrainDataset

# from types import SimpleNamespace
# from torch.utils.data import DataLoader
# # -----------------------------
# # Create a dummy args object with necessary attributes.
# args = SimpleNamespace(
#     patch_size=256,
#     data_file_dir="/mnt/c/Users/nuoxi/year4Code/fyp/InstructIR/train_data_names/",
#     # For example purposes
#     denoise_dir="/mnt/c/Users/nuoxi/year4Code/fyp/InstructIR/data/Train/denoise/",
#     derain_dir="/mnt/c/Users/nuoxi/year4Code/fyp/InstructIR/data/Train/derain/",
#     de_type=["derain", "denoise_15"]
# )

# dataset = InstructIRTrainDataset(args)

# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# for batch in dataloader:
#     # The __getitem__ returns a tuple: ([clean_name, deg_id], degrad_patch, clean_patch)
#     meta_info, degrad_patch, clean_patch = batch
#     print("Meta Info:", meta_info, "\n\n\n")
#     print("Meta Info len:", len(meta_info[0]), "\n\n\n")
#     print("Degraded patch tensor shape:", degrad_patch.shape, "\n\n\n")
#     print("Clean patch tensor shape:", clean_patch.shape, "\n\n\n")
#     break  # only process one batch for testing

import os
print(os.cpu_count())
