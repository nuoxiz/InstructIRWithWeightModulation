# Image Restoration Using Weight Modulation

**2024-25 BA Integrated Computer Science Final Year Project**  
**Nuoxi Zhang**

## Acknowledgement

This project is built on top of the InstructIR model developed by Conde et al. (2024).

- **ArXiv:** [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2401.16468)
- **GitHub:** [InstructIR Repository](https://github.com/mv-lab/InstructIR)
- **Hugging Face Demo:** [![Hugging Face](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/marcosv/InstructIR)

## Instructions

You can try out the code using one of the two options below:

### Option 1: Google Colab

1. Navigate to the `colabNotebook` folder in this repository.
2. Upload the two Colab notebooks to your Google Colab account.
3. Follow the instructions in **Section 2** for setting up your Google Drive.

### Option 2: Local Setup

1. Clone this repository to your local machine.
2. Follow the setup instructions provided in **Section 3** below.

### Download Datasets pretrained weights

- **Test Dataset:** [Download Link](https://drive.google.com/file/d/1_Lwp-wpRyigWBL0QywsJhlX04e7T6ajS/view?usp=sharing)
- **Training Dataset:** [Download Link](https://drive.google.com/file/d/1Vz1yQ9K74HO2_G0wD0qWOB3f7IOPmhyo/view?usp=sharing)
- **Pretrained weights** [Download Link](https://drive.google.com/drive/folders/1m_DW5RJ_EssLOJ8yrRfEtvHTfhiq4GOj?usp=sharing)

---

## Section 2: Google Colab Setup

Before running the Colab notebooks, configure your Google Drive as follows:

1. **Create a Parent Folder:**  
   In your Google Drive, create a new folder named `FYPData`.

2. **Upload the Models:**  
   Copy the entire `models/` folder from this repository into the `FYPData` folder.

3. **Test Dataset:**

   - Download and unzip the test dataset.
   - Upload the unzipped test dataset to `FYPData` and rename the folder to `test-data`.

4. **Training Dataset:**

   - Download and unzip the training dataset.
   - Upload the unzipped training dataset to `FYPData` and rename the folder to `Train`.

5. **Additional Files:**
   - Upload the `train_data_names/` folder from this repository to `FYPData`.
   - Upload the file `text/human_instructions.json` to `FYPData`.

---

## Section 3: Local Setup

To run the training and testing code on your local machine, follow these steps:

1. **Download and Unzip Datasets:**  
   Download both the test and training datasets and the pretrained weights using the provided links and unzip them.

2. **Organize the Training Dataset:**  
   Ensure that the training dataset is structured as shown below (create folders manually if needed):

./data  
└── Train  
 ├── dehaze  
 │ ├── original  
 │ └── synthetic  
 ├── denoise  
 └── derain  
 ├── original  
 └── rainy

3. **Organize the Test Dataset:**  
   Ensure that the test dataset is structured as follows (create folders manually if needed):

./test-data  
├── denoising_testsets  
│ ├── CBSD65  
│ ├── CBSD65_15  
│ ├── CBSD65_25  
│ └── CBSD65_50  
├── Kodak24  
│ ├── Kodak24_15  
│ ├── Kodak24_25  
│ └── Kodak24_50  
├── Rain100L  
│ ├── original  
│ └── rainy  
└── SOTS  
 ├── GT  
 └── IN  
3. **Run**

```python
python requirements_fyp.txt
python fyp_eval.py
python fyp_train.py
```
