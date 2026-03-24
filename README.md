# VP-VLA: Visual Prompting as an Interface for Vision-Language-Action Models

<p align="center">
  <a href="https://visualprompt-vla.github.io/">
    <img src="https://img.shields.io/badge/Project%20Page-blue?style=for-the-badge&logo=homeassistant&logoColor=white" alt="Project Page">
  </a>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2603.22003">
    <img src="https://img.shields.io/badge/Paper%20(arXiv)-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper (arXiv)">
  </a>
</p>


> Vision-Language-Action (VLA) models often struggle with precise spatial grounding and robustness due to monolithic end-to-end designs. In this project, we introduce that decouples high-level reasoning and low-level execution via a structured visual prompting interface, enabling more precise and reliable robotic manipulation.  

---


https://github.com/user-attachments/assets/9b17ca16-3fba-4962-9116-6cf27e8242fa



## Overview of VP-VLA

<div align=center>
<img width="98%" src="assets/teaser.png"/>
</div>

VP-VLA demonstrates the following features:

1. **Dual-System Architecture**: VP-VLA decomposes robotic manipulation into:
   - **System 2 Planner** (high-level reasoning)
   - **System 1 Controller** (low-level execution)  

2. **Visual Prompt Interface**: Instead of relying solely on text, VP-VLA converts language instructions into **structured visual prompts** (crosshairs and bounding boxes), enabling precise spatial grounding.  

3. **Improved Spatial Precision & Robustness**: By grounding actions in visual space, the framework significantly improves performance in:
   - Novel object scenarios  
   - Out-of-distribution (OOD) spatial configurations  

4. **General Multi-Stage Manipulation**: VP-VLA supports complex, multi-step tasks via:
   - Task decomposition  
   - Event-driven planning  
   - Dynamic visual prompt updates  

---

## News

[Mar 24th, 2026] 🔥 [📖 Paper](https://arxiv.org/abs/2603.22003) released! Code will be released within two weeks.  

---

## Contents
- [Model](#model)
- [Installation](#installation)
- [Evaluation](#evaluation)
- [Training](#training)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

---

## Model

<div align=center>
<img width="98%" src="assets/pipeline.png"/>
</div>

VP-VLA consists of two key components:

### System 2 Planner (High-Level Reasoning)
- Decomposes instructions into subtasks
- Identifies:
  - Target object  
  - Target location  
- Generates structured visual prompts  

### System 1 Controller (Low-Level Execution)
- Takes:
  - Original observation  
  - Visual prompt overlay  
- Produces:
  - Continuous robot actions  

### Key Idea
Instead of solving everything in one forward pass, VP-VLA does the following:
- **Language → Visual Prompts → Actions**

This transforms the problem into **visuomotor tracking of explicit spatial cues**, improving precision and interpretability.  

---

## Citation
```bibtex
@article{wang2026vpvla,
  title={VP-VLA: Visual Prompting as an Interface for Vision-Language-Action Models},
  author={Wang, Zixuan and Chen, Yuxin and Liu, Yuqi and Ye, Jinhui and Chen, Pengguang and Lu, Changsheng and Liu, Shu and Jia, Jiaya},
  journal={arXiv preprint arXiv:2603.22003},
  year={2026}
}
```
## Acknowledgement
We would like to thank the following repos for their great work:
- This work is built upon [starVLA](https://github.com/starVLA/starVLA)
- This work utilizes models from [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) and [SAM3](https://huggingface.co/facebook/sam3) 
