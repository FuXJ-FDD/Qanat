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
Xingjian Fu, Lei Luo, Feng Li, Jia Yang, Jie Shao, Ran Tu, Jinhui Fan, Zhihong Luo, Zhi Zhang, Deep learning with geographical post-processing optimization: an integrated framework for detecting qanat activity states, Journal of Archaeological Science, Volume 188, 2026, 106526, ISSN 0305-4403, https://doi.org/10.1016/j.jas.2026.106526.
**Abstract: **As ancient underground water systems that sustained civilizations in arid regions for millennia, qanats represent both remarkable hydraulic heritage and vital water sources, with the Persian Qanat (inscribed on the World Heritage List in 2016) requiring dynamic monitoring for effective protection and management. This study overcomes limitations of prior spatial-distribution-focused research by constructing the first multi-region annotated dataset from very high-resolution resolution Google Earth satellite imagery across Iran, Afghanistan, Morocco and China, classifying 8,587 active and 17,383 inactive qanat samples. Our YOLO11-based model (enhanced with C3k2 backbone and C2PSA attention) integrates a novel post-processing framework where DBSCAN clustering removed 90.8% of outliers – collectively achieving 97.16% precision (9.5% improvement over baseline) and 76.56% recall. Applied to 11 Persian Qanat World Heritage Sites, the system identified 41,781 shafts in 889 qanats, including 15,742 active and 26,039 inactive qanats, revealing key patterns: 6/km2 density, 169 m (SD = 46.3 m) spacing, and 95% occurrence in bare/sparsely vegetated areas on gentle slopes (mean 2.5°). This high-precision dataset enables prioritized conservation of inactive qanats as cultural relics and sustainable management of active systems, demonstrating how AI-geospatial integration can revolutionize archaeological monitoring in arid landscapes.
**Keywords: **Qanat activity states; Cultural heritage monitoring; Remote sensing; Deep learning; Post-processing optimization
