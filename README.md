# U‑Net-PyTorch

An **end-to-end, under-the-hood** implementation of **U-Net** for **binary image segmentation**—starting with **preparing images and masks**, moving through the **encoder–decoder with skip connections** and **loss/metric design**, and finishing with **training, evaluation, and crisp visualizations**. Built in **PyTorch** for a clean, **reproducible pipeline you can read, tweak, and reuse**. The notebook tackles **brain-tumor segmentation**—a practical, challenging medical-imaging task with **subtle boundaries** and **class imbalance**.


---

## Repository Structure

```
U‑Net‑PyTorch/
├── U‑Net/                          # directory containing the notebook and checkpoint
│ ├── U‑Net.ipynb                   # end-to-end pipeline notebook (brain-tumor segmentation)
│ │ ├── Dataset loading & preprocessing
│ │ ├── U‑Net architecture definition
│ │ ├── Loss & optimizer setup
│ │ ├── Training loop with logging & checkpointing
│ │ ├── Dice coefficient evaluation
│ │ └── Visualization of predictions vs. ground truth
│ └── checkpoint.md                 # checkpoint containing epoch, model_state_dict, optimizer_state_dict, lr_scheduler_state_dict, loss, and dice_score from the best epoch
│     ├── epoch                      
│     ├── model_state_dict         
│     ├── optimizer_state_dict      
│     ├── lr_scheduler_state_dict  
│     ├── loss                      
│     └── dice_score                
├── segmentation_comparison.png     # overlay: image + ground truth mask | image + predicted mask
├── mask_comparison.png             # side‑by‑side: image | ground truth | prediction
├── LICENSE                         # MIT License text
└── README.md                       # this document
```

---
## Scores Achieved  

**Benchmark run** — trained for **64 epochs**, results:  

- **Train**  
  - *Dice Score*: ***0.93***
  - *Loss*: ***0.0948***

- **Test**  
  - *Dice Score*: ***0.80***
  - *Loss*: ***0.2422***

**Trained weights are included at** `U‑Net/checkpoint.pth`.

---
## Sample Segmentation Results

Below are two example outputs produced by the U-Net pipeline while being tested:



#### Comparison between the output and the ground truth segemntation in testing  

![Segmentation Comparison](https://github.com/franciszekparma/U-Net-PyTorch/blob/162be42e858d2cc66024425f5293f52a38bbb23e/segmentation_comparison.png)



#### Comparison between the predicted mask by the model and the ground truth mask in testing  

![Mask Comparison](https://github.com/franciszekparma/U-Net-PyTorch/blob/aa32f4b3cc8450f17b6bf56eaa12b6467fce363c/mask_comparison.png)


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

---

## Dataset Format

Expected layout (customize paths in the notebook if needed):

```
DATASET_ROOT/
├── segmentation_task/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
...
```

### **Notes:**
• Masks are interpreted as binary; if stored as {0, 255}, they are normalized to {0, 1}.  
• Images are resized/normalized in transforms; keep image size consistent across training/eval.  
• For imbalanced foreground, consider stronger augmentation or changing the loss / adding a weight to a paritucal part of the loss.  

---

## Training (Reference Setup)

* Epochs: **64** (reference)
* Metric: **Dice coefficient** (reported on train/test)
* Loss:   **BCEWithLogitsLoss + Dice**
* Checkpointing: best weights saved to `checkpoint.pth`  

### **Tips:**

• Start with a moderate image size if GPU memory is limited. 
• Use stronger data augmentation techniques (Horizontal Flip, ShiftScaleRotate, Blur, etc.)  
• Use the Albumentations library for image augmentation (strongly recommended)
• Monitor Dice and loss together; verify thresholds used for binarization.  
• Save `state_dict` for portability.    

---

## Troubleshooting

• **CUDA out of memory** → reduce batch size or image size; ensure tensors are moved off GPU when not needed.  
• **All‑black or all‑white outputs** → check mask normalization and loss/thresholding.  
• **Tensor size mismatch on skip connections** → verify resize/crop/stride consistency. 


---
## The Most Important Tip  
• **Experiment with the code!** This is the best way to understand / learn all the code / theory related to the given topic.  

---
## Contributing

Issues and PRs are welcome — bug fixes, training tips, alternative losses (Focal/Tversky/...), multi‑class extensions, documentation improvements, other improvements to the implementation.


---
## Acknowledgements

Ronneberger, Fischer, Brox — *U‑Net: Convolutional Networks for Biomedical Image Segmentation* (MICCAI 2015)

---

## License

This project is licensed under the MIT License.  
© *franciszekparma* 
