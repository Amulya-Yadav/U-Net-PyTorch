# U‑Net-PyTorch

A fast, compact implementation of **U‑Net** for binary semantic segmentation in PyTorch. Clean pipeline for training, evaluation, and clear visualizations.

---

## Repository Structure

```
U‑Net‑PyTorch/
├── U‑Net/                          # directory containing the notebook and checkpoint
│ ├── U‑Net.ipynb                   # end‑to‑end pipeline notebook
│ │ ├── Dataset loading & preprocessing
│ │ ├── U‑Net architecture definition
│ │ ├── Loss & optimizer setup
│ │ ├── Training loop with logging & checkpointing
│ │ ├── Dice coefficient evaluation
│ │ └── Visualization of predictions vs. ground truth
│ └── checkpoint.md                 # checkpoint containing epoch, model_state_dict, optimizer_state_dict, lr_scheduler_state_dict, loss, and dice_score from the best epoch
│     ├── epoch                     # the epoch with the best dice score 
│     ├── model_state_dict          # model weights
│     ├── optimizer_state_dict      # optmizer info (contains everything PyTorch needs to resume training with the exact same optimizer state as when it was saved.)
│     ├── lr_scheduler_state_dict   # contains info such that you can resume the schedule exactly where it left off
│     ├── loss                      # saved loss value with the best dice score
│     └── dice_score                # the best dice score achieved by the model while training
├── segmentation_comparison.png     # overlay: image + ground truth mask | image + predicted mask
├── mask_comparison.png             # side‑by‑side: image | ground truth | prediction
├── LICENSE                         # MIT License text
└── README.md                       # this document
```

---

## Highlights

• Classic U‑Net with skip connections (encoder/decoder)  
• Binary mask head (1 channel) with sigmoid at inference  
• Dice as the primary metric; BCEWithLogitsLoss commonly used  
• Minimal utilities for overlays and mask comparisons  
• Notebook‑first workflow for quick iteration  

---

## Sample Segmentation Results

Below are two example outputs produced by the U-Net pipeline while being tested:

![Segmentation Comparison]([https://github.com/franciszekparma/YOLOv1-PyTorch/blob/57fb191d9d4beee2dbec3a4bef721fbcf873ea2c/sheep.png](https://github.com/franciszekparma/U-Net-PyTorch/blob/main/segmentation_comparison.png?raw=true))

![Mask Comparison]([https://github.com/franciszekparma/YOLOv1-PyTorch/blob/57fb191d9d4beee2dbec3a4bef721fbcf873ea2c/biker.png](https://github.com/franciszekparma/U-Net-PyTorch/blob/main/mask_comparison.png?raw=true))

**Reference run:** trained for **64 epochs** 
· **Dice 0.93 (train)** 
· **Dice 0.80 (test)**.
Trained weights are included at `U‑Net/checkpoint.pth`.

---

## Prerequisites

• Python 3.8+  
• pip  
• (Optional) CUDA‑enabled GPU  

Install packages:

```
pip install torch torchvision numpy matplotlib pillow tqdm
# optional
pip install opencv-python albumentations jupyterlab
```

---

## Getting Started

* Clone the repository  

```
git clone https://github.com/franciszekparma/U-Net-PyTorch.git
cd U-Net-PyTorch
```

* Launch Jupyter  

```
jupyter lab   # or: jupyter notebook
```

* Open and run `U‑Net/U‑Net.ipynb`

Set your image and mask folders, adjust batch size / image size to your GPU memory, run all cells to train and evaluate.

---

## Dataset Format

Expected layout (customize paths in the notebook):

```
DATASET_ROOT/
├── images/   # input images (grayscale or RGB)
└── masks/    # binary masks (0/1 or 0/255)
```

Notes:

• Masks are interpreted as binary; if stored as {0, 255}, they are normalized to {0, 1}.
• Images are resized/normalized in transforms; keep image size consistent across training/eval.
• For imbalanced foreground, consider stronger augmentation or changing the loss / add a weight to a paritucal part.

---

## Training (Reference Setup)

* Epochs: **64** (reference)
* Metric: **Dice coefficient** (reported on train/test)
* Loss: commonly **BCEWithLogitsLoss + Dice**
* Checkpointing: best weights saved to `U‑Net/checkpoint.pth`  

Tips:

• Start with a moderate image size if GPU memory is limited.  
• Monitor Dice and loss together; verify thresholds used for binarization.  
• Save `state_dict` for portability.  

---

## Troubleshooting

• **CUDA out of memory** → reduce batch size or image size; ensure tensors are moved off GPU when not needed.  
• **All‑black or all‑white outputs** → check mask normalization and loss/thresholding.  
• **Tensor size mismatch on skip connections** → verify resize/crop/stride consistency. 

---

## Roadmap

• Optional multiclass head and Cross‑Entropy loss path
• More augmentation presets
• Light inference script (CLI)
• Metric suite: IoU, precision/recall, PR curves

---

## Contributing

Issues and PRs are welcome — bug fixes, training tips, alternative losses (Focal/Tversky), multi‑class extensions, documentation improvements, other improvments to the implementation.

---

## Acknowledgements

Ronneberger, Fischer, Brox — *U‑Net: Convolutional Networks for Biomedical Image Segmentation* (MICCAI 2015)

---

## License

This project is licensed under the MIT License.  
© franciszekparma  
