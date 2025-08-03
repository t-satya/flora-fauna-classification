# 🐾 Flora & Fauna Image Classification

This repository contains experiments with various deep learning architectures for multi-class image classification of flora and fauna species. The goal was to identify the best-performing model architecture and training strategy using PyTorch and the `timm` library.

## 📊 Dataset
- Images of various plant and animal species.
- Multi-class classification task.
- Evaluation metric: **F1 score (Weighted)**

## 🧪 Models & Results

### ✅ 1. EfficientNet (CNN-based)
- Model: `efficientnet_v2_s`
- Training: All parameters updated (fine-tuned)
- Augmentation: Custom
-  Optimizer: `Adam`
- **Best Validation F1 Score:** `0.9069`

### ✅ 2. ConvNeXt
- Model: `convnext_base`
- Training: Only the `head` was unfrozen and trained (feature extraction)
- Optimizer: `AdamW`
- Augmentation: Rand-M9 policy + Bicubic interpolation
- **Best Validation F1 Score:** `0.9280` (Best overall)

### ✅ 3. Vision Transformer (ViT)
- Model: `vit_base_patch16_224`
- Training: Last few transformer blocks + classification head were unfrozen
- Optimizer: `AdamW`
- Augmentation: Same as ConvNeXt
- **Best Validation F1 Score:** `0.9191`

---

## 🔧 Libraries & Tools
- PyTorch
- `timm` (pretrained models & augmentations)
- AMP (Automatic Mixed Precision for faster training)
- Custom training loop with learning rate scheduling and early stopping

---

## 📁 Notebooks
| Model        | Notebook Name                       | Notes                             |
|--------------|-------------------------------------|-----------------------------------|
| EfficientNet | `image_classification_efficientNet_b1.ipynb`   | Best CNN model with full fine-tuning |
| ConvNeXt     | `convnext_best_model_v1.ipynb`       | Best model overall (head only)   |
| ViT Base     | `ViT_base_v2.ipynb`      | Transformer-based model          |

---

## 📝 Notes
- All models were trained on 224×224 images.
- ConvNeXt significantly benefited from freezing the backbone and only training the classification head.
- ViT required partial unfreezing to achieve good generalization.
- Used F1 score (Weighted).

---

## 🚀 Future Work
- Try larger ViT variants (e.g., `vit_large` or `vit_huge`) with partial fine-tuning.
- Explore Swin Transformers and Hybrid architectures.
- Optimize data augmentation for ViT further.

---

## 📌 Author
Satya Takkellapati  
Aspiring Data Scientist | Deep Learning & Computer Vision Enthusiast  

