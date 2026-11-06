# 3D Reconstruction of the Visible Human Project via Deep Learning-based Segmentation

A complete pipeline for the three-dimensional reconstruction of anatomical structures from the **Visible Human Project (VHP)** dataset, utilizing semantic segmentation with **U-Net**, 3D mesh reconstruction, and skin texture reprojection.

Project developed as a final-year undergraduate thesis (TCC) in Computer Science at the State University of Santa Cruz (UESC).

---

## 📋 Overview

The pipeline transforms a sequence of **1,865 2D images** from the VHP dataset into a **textured 3D model** through three main stages:

1. **Semantic segmentation** of the slices using a trained U-Net, generating binary masks.
2. **3D mesh reconstruction** from the masks via the Marching Cubes algorithm.
3. **RGB texture (skin) reprojection** onto the reconstructed mesh.

Result: a 3D model with approximately **2.27 million vertices** and **4.54 million faces**.

---

## 🧠 Segmentation (U-Net)

- Trained model: `2026-05-18_Unet_visible_best.h5`
- Applied to the 1,865 PNG images from the VHP dataset.
- Evaluated using the **Dice coefficient** and **IoU (Intersection over Union)**.
- Binary masks (rather than the segmented color images) were used as input for the reconstruction; using the color images directly resulted in unwanted internal vertices (due to the segmentation of internal organs) and memory overflow, which prevented the mesh from being properly closed. ---

## 🧊 3D Reconstruction — `reconstrucao_3d_vhp.py`

Reconstructs a 3D mesh from **background-removed binary images** (segmented masks, with the original image background excluded).

**Key Parameters:**

| Parameter | Value | Description |
|---|---|---|
| `FATOR` | 0.25 | Image downscaling factor |
| `SPACING_XY` | 0.33 mm/px | Pixel spacing in the XY plane (value adjusted based on testing) |
| `SPACING_Z` | 1.0 mm | Spacing between slices along the Z-axis |
| `CHUNK_SIZE` | 64 | Block size for chunk-based Marching Cubes |
| `OVERLAP` | 16 | Overlap between chunks |

**Steps:**
- Isotropic interpolation via `scipy.ndimage.zoom`
- Chunk-based Marching Cubes (block-wise processing to prevent memory overflow)
- Smoothing via Laplacian smoothing
- Export: `model3d.ply`

---

## 🎨 Texturing — `aplicar_textura_pele.py` (v7)

Applies RGB skin texture onto the reconstructed mesh, using the alpha channel of the transparent PNG images as a mask.

**Key Parameters:**

| Parameter | Value | Description |
|---|---|---|
| `EROSAO_BORDA` | 6 px | Morphological erosion applied to the edges of the alpha channel |
| `ESPESSURA_CASCA` | 20 px | Thickness of the skin "shell" via distance transform |
| `JANELA_BLEND` | ±1 slice | Blending window between adjacent slices | **Output:** `model3d_texturizado_v7.ply` — a model with preserved texture details (e.g., an American flag tattoo).

> Note: The solid model was exported without vertex colors; the hollow model retains the skin texture and visual details.

---

## 🔧 Technical issues resolved

During development, several engineering obstacles were overcome:

- Windows "long path" limitations causing TensorFlow to fail
- Unicode errors in file paths
- Progress bars freezing during long processing tasks
- Axis-mapping bugs during texture reprojection
- OpenCV failures when handling paths with accented characters (resolved by using Pillow and short file paths)
- Detection of duplicate files on Windows
- VHP XY-spacing adjusted to 0.33 mm/px based on testing results
- Changes to the `trimesh` API
- File-locking issues with `numpy.memmap` on Windows

---

## 📈 Project evolution

The current pipeline is the result of several iterations:

1. **3D Slicer** (Threshold → Islands → Smoothing → Create Closed Surface → Wrap Solidify → PLY export → visualization in MeshLab), utilizing a Python script for "color-baking" that maps OBJ vertex coordinates to pixels in PNG images.
2. **Delta Rule (RGBDZ)** — pixel-by-pixel subtraction between consecutive VHP slices to extract surface information, replicating an approach from previous work at UESC (Bomfim, 2017).
3. A prototype developed in **Google Colab** prior to migrating to a local Windows environment.
4. The current pipeline, implemented entirely in Python, featuring segmentation via U-Net and reconstruction using the Marching Cubes algorithm. ---

## 📚 Related work

Compared to previous 3D reconstruction studies using the VHP dataset (Bonfim, 2017; Bastos, 2020; Campos, 2022), this pipeline:

- Utilizes the **complete set of images**, with no need for data reduction
- Produces **more detailed** models
- Avoids the quality limitations observed in previous studies

---

## 🚀 Future directions

- More recent segmentation architectures: **UNet++**, **3D UNet** (Zhou et al., 2018)
- Related applications in remote sensing (LeCun; Bengio; Hinton, 2015)
- Exploration of **Attention U-Net** and **SAM (Segment Anything Model)** as state-of-the-art approaches in segmentation

---

## 🛠️ Technical stack

`Python` · `TensorFlow/Keras` · `OpenCV` · `scipy` · `trimesh`
