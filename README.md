# Code for NAPG: Neighborhood-Assisted Multi-Prototype Group Model for Cross-Domain Remote Sensing Image Semantic Segmentation  

## Overview  

This repository contains the implementation of **NAPG**, a Neighborhood-Assisted Multi-Prototype Group Model designed for cross-domain semantic segmentation of remote sensing images. The model introduces a novel multi-stage pipeline that integrates prototype-based learning and spatial topological consistency to address the challenges of domain adaptation.  

The repository is structured into three main stages, each encapsulated in its own script:  

---

### 1. Step 1: Warm-Up Training  
**File:** `step1-warm-up.py`  

This script performs **warm-up training** using single-prototype group representation and domain adversarial training.  

- **Core Components:**  
  - **Data Loaders:** Uses `CreateDataLoader` to load source and target datasets.  
  - **Model Architecture:** Utilizes the `DeepLabV3PlusSimGlobalLinearKL3neibor` model for semantic segmentation.  
  - **Loss Functions:**  
    - Cross-entropy loss for supervised segmentation.  
    - Entropy-based adversarial loss for domain adaptation.  
  - **Prototype Representation:** Generates single-prototype groups for the source domain using k-means clustering.  
  - **Entropy-Based Adversarial Module:** Aligns marginal distributions between source and target domains.  

- **Training Workflow:**  
  1. Load source and target datasets.  
  2. Train the model on source domain data using prototype features.  
  3. Use adversarial training to align source and target domain distributions.  
  4. Periodically validate to assess prototype-based segmentation performance.  

---

### 2. Step 2: UDA Training  
**File:** `step2-UDA-training.py`  

This script focuses on **UDA training** by generating multi-prototype groups (MPG) and optimizing with neighborhood consistency.  

- **Core Components:**  
  - **Multi-Prototype Group Generation:** Uses high-confidence target samples to create diverse prototype groups.  
  - **Neighborhood Consistency Loss:** Enhances prototype optimization by considering spatial topological information.  
  - **Loss Functions:**  
    - Variance loss for spatial consistency.  
    - Cross-entropy loss for segmentation performance.  

- **Training Workflow:**  
  1. Generate multi-prototype groups based on high-confidence target samples.  
  2. Train the model using both source and target domain data, incorporating MPG.  
  3. Refine prototypes with neighborhood consistency estimation (GNCE).  

---

### 3. Step 3: Teacher-Student Learning， not in paper！
**File:** `step3-teacher-student.py`  

This script implements a **teacher-student framework** to further refine the model for UDA.  

- **Core Components:**  
  - **Teacher-Student Models:** Uses a pretrained teacher model to guide the student model.  
  - **Knowledge Distillation:** Transfers knowledge from teacher to student using a KL divergence loss.  
  - **Prototype-Based Refinement:** Incorporates prototype features from both source and target domains.  

- **Training Workflow:**  
  1. Initialize the teacher model with pretrained weights from Step 2.  
  2. Train the student model using both supervised segmentation loss and knowledge distillation loss.  
  3. Validate the student model to assess improvements in cross-domain adaptation.  

---

## Experimental Setup  

### Training Environment  
- **GPU:** NVIDIA 3090 (24GB memory)  
- **Batch Size:** 10  
- **Learning Rate:** 0.001

### Dataset Configurations  

To evaluate the UDA performance of the proposed NAPG model, four experimental configurations were used:  

1. **POT-RGB → VAI:**  
   - **Source Domain:** RGB bands from the POT dataset.  
   - **Target Domain:** VAI dataset.  
   - **Challenge:** Different imaging locations, spatial resolutions, and band compositions.  

2. **POT-IRRG → VAI:**  
   - **Source Domain:** IRRG bands from the POT dataset.  
   - **Target Domain:** VAI dataset.  
   - **Challenge:** Different imaging locations and spatial resolutions but identical band compositions.  

3. **VAI → POT-IRRG:**  
   - **Source Domain:** VAI dataset.  
   - **Target Domain:** IRRG bands from the POT dataset.  
   - **Challenge:** Same band compositions but different imaging locations and spatial resolutions.  

4. **VAI → POT-RGB:**  
   - **Source Domain:** VAI dataset.  
   - **Target Domain:** RGB bands from the POT dataset.  
   - **Challenge:** Different imaging locations, band compositions, and spatial resolutions.  

---

## Workflow  

1. **Step 1: Warm-Up Training**  
   Run `step1-warm-up.py` to initialize the model with single-prototype groups and align marginal distributions.  

2. **Step 2: UDA Training**  
   Execute `step2-UDA-training.py` to generate multi-prototype groups and optimize them using neighborhood consistency.  

3. **Step 3: Teacher-Student Learning**  
   Apply `step3-teacher-student.py` for further refinement using the teacher-student framework.  

---

## Citation  

