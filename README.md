This repository contains the code and trained models for the research on **Qanat detection and activity status identification**.

## 📦 Dataset (14.7GB)
Due to the large file size, the full training dataset (images and labels) has been uploaded to **Zenodo** to comply with open-access requirements and peer review.

* **Dataset Link:** [https://doi.org/10.5281/zenodo.18636032](https://doi.org/10.5281/zenodo.18636032)
* **DOI:** 10.5281/zenodo.18636032

## 🚀 Model Weights
The trained YOLO11 weights are provided in this repository for verification:
- `weights/best.pt`: The best-performing model checkpoint.
- `weights/last.pt`: The final checkpoint after training.

## 📄 Related Publication
If you find this repository, code, or dataset useful for your research, please consider citing our published paper:

[cite_start]**Fu, X.**, Luo, L., Li, F., Yang, J., Shao, J., Tu, R., Fan, J., Luo, Z., & Zhang, Z. [cite: 8, 9] (2026) [cite_start][cite: 2]. [cite_start]Deep learning with geographical post-processing optimization: an integrated framework for detecting qanat activity states[cite: 7]. [cite_start]*Journal of Archaeological Science* [cite: 4][cite_start], 188, 106526[cite: 2]. [cite_start][https://doi.org/10.1016/j.jas.2026.106526](https://doi.org/10.1016/j.jas.2026.106526) [cite: 36]

> **📌 Important Notes for Users:**
> * **Manuscript Reference:** When reviewing the evaluation metrics and findings in the manuscript, please ensure you refer to **Table 4** for the correct data summary (an earlier typesetting draft contained a misprint referring to Table 6).
> * **Dataset Extraction Scope:** For researchers utilizing our global-scale vector extraction data, please note that the technical extraction findings in **South Korea, Qatar, Bangladesh, and India** have been identified as false positives (primarily roads or river features) and do not represent actual Qanats.
> * **Vegetation Analysis Scripts:** If you are running our batch processing scripts for vegetation monitoring around the Qanat systems, please note that the analysis index logic has been updated to use **kNDVI** instead of EVI.
