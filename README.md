# <p align=center>Awesome Multimodal Large Language Models In Low-level Vision[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/ChunmingHe/awesome-multimodal-large-language-models-in-low-level-vision)</p>

<p align=center>🔥A curated list of awesome <b>Multimodal Large Language Models(MLLMs)</b> & <b>Vision-Language Models(MLLMs)</b> in low-level vision.🔥</p>

<p align=center>Please feel free to offer your suggestions in the Issues and pull requests to add links.</p>

<p align=center><b>[ Last updated at 2026/01/18 ]</b></p>

## Contents

- [Awesome Multimodal Large Language Models In Low-level Vision](#awesome-multimodal-large-language-models-in-low-level-vision)
  - [Contents](#contents)
  - [Latest Works Recommended](#latest-works-recommended)
  - [Awesome Papers](#awesome-papers)
    - [1. Direct VLM Adaptation for Low-Level Vision](#1-direct-vlm-adaptation-for-low-level-vision)
      - [1.1 Visual Encoder Adaptation: Handling Details](#11-visual-encoder-adaptation-handling-details)
        - [1.1.1 Resolution Scaling](#111-resolution-scaling)
        - [1.1.2 Feature Fusion](#112-feature-fusion)
      - [1.2 Language Branch Adaptation: Bridging Modalities](#12-language-branch-adaptation-bridging-modalities)
        - [1.2.1 Prompt Learning Strategies](#121-prompt-learning-strategies)
        - [1.2.2 Instruction Tuning for Restoration](#122-instruction-tuning-for-restoration)
      - [1.3 Output Head Adaptation: From Tokens to Pixels](#13-output-head-adaptation-from-tokens-to-pixels)
        - [1.3.1 Tokenizer–Decoder Framework](#131-tokenizerdecoder-framework)
      - [1.4 Parameter-Efficient Fine-Tuning in Restoration](#14-parameter-efficient-fine-tuning-in-restoration)
        - [1.4.1 LoRA & Adapter Integration](#141-lora--adapter-integration)
        - [1.4.2 Freezing Strategies](#142-freezing-strategies)
    - [2. VLM as Auxiliary for Low-Level Vision](#2-vlm-as-auxiliary-for-low-level-vision)
      - [2.1 VLM as Semantic Provider: Text-Guided Restoration](#21-vlm-as-semantic-provider-text-guided-restoration)
        - [2.1.1 Text-Conditioned Injection](#211-text-conditioned-injection)
        - [2.1.2 Language-Driven Manipulation](#212-language-driven-manipulation)
        - [2.1.3 Subject-Aware Restoration](#213-subject-aware-restoration)
      - [2.2 VLM as Degradation Interpreter: Visual-Prompting & Context](#22-vlm-as-degradation-interpreter-visual-prompting--context)
        - [2.2.1 Degradation Classification](#221-degradation-classification)
        - [2.2.2 Description-based Restoration](#222-description-based-restoration)
      - [2.3 VLM as Quality Evaluator: Perception and Assessment](#23-vlm-as-quality-evaluator-perception-and-assessment)
        - [2.3.1 No-Reference Quality Assessment](#231-no-reference-quality-assessment)
        - [2.3.2 Semantic Consistency Loss](#232-semantic-consistency-loss)
        - [2.3.3 Feedback Loops](#233-feedback-loops)
      - [2.4 VLM as Intelligent Controller: Agent-Based Frameworks](#24-vlm-as-intelligent-controller-agent-based-frameworks)
        - [2.4.1 Motivation as Intelligent Controller](#241-motivation-as-intelligent-controller)
        - [2.4.2 Tool Usage & Orchestration](#242-tool-usage--orchestration)
        - [2.4.3 Iterative Refinement](#243-iterative-refinement)
    - [3. Extended Applications](#3-extended-applications)
      - [3.1 Medical Image Processing](#31-medical-image-processing)
        - [3.1.1 Biomedical VLMs](#311-biomedical-vlms)
        - [3.1.2 CT and MRI](#312-ct-and-mri)
      - [3.2 Remote Sensing Data Processing](#32-remote-sensing-data-processing)
        - [3.2.1 Spatial-domain Tasks](#321-spatial-domain-tasks)
        - [3.2.2 Spectral-domain Tasks](#322-spectral-domain-tasks)
      - [3.3 Other Extended Applications](#33-other-extended-applications)
        - [3.3.1 CAD](#331-cad)
        - [3.3.2 Video Processing Tasks](#332-video-processing-tasks)
        - [3.3.3 3D Processing Tasks](#333-3d-processing-tasks)
  - [Datasets](#datasets)
  - [Related Surveys Recommended](#related-surveys-recommended)
  - [Reference](#reference)

## <a id="latest-works-recommended">Latest Works Recommended</a>

**Diffusion Models in Low-Level Vision: A Survey**<br />*Chunming He, Yuqi Shen, Chengyu Fang, Fengyang Xiao, Longxiang Tang, Yulun Zhang, Wangmeng Zuo, Zhenhua Guo, Xiu Li*<br />TPAMI. [[Paper](https://arxiv.org/abs/2406.11138)] 

**Reti-Diff: Illumination Degradation Image Restoration with Retinex-based Latent Diffusion Model**<br />*Chunming He, Chengyu Fang, Yulun Zhang, Kai Li, Longxiang Tang, Chenyu You, Fengyang Xiao, Zhenhua Guo, Xiu Li*<br />
ICLR 2025, Spotlight. [[Paper](https://arxiv.org/abs/2311.11638)] [[Github](https://github.com/ChunmingHe/Reti-Diff)]<br />
Jan. 2025<br />

## <a id="awesome-papers">Awesome Papers</a>

### <a id="1-direct-vlm-adaptation-for-low-level-vision">1. Direct VLM Adaptation for Low-Level Vision</a>

#### <a id="11-visual-encoder-adaptation-handling-details">1.1 Visual Encoder Adaptation: Handling Details</a>

##### <a id="111-resolution-scaling">1.1.1 Resolution Scaling</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/om-ai-lab/ImageRAG.svg?style=social&label=Star) <br> [**Enhancing Ultra High Resolution Remote Sensing Imagery Analysis with ImageRAG**](https://arxiv.org/abs/2411.07688) <br> | GRSM | 2024-11 | [Github](https://github.com/om-ai-lab/ImageRAG) |
| ![Star](https://img.shields.io/github/stars/NVlabs/FeatSharp.svg?style=social&label=Star) <br> [**FeatSharp: Your Vision Model Features, Sharper**](https://arxiv.org/abs/2502.16025) <br> | ICML | 2025-02 | [Github](https://github.com/NVlabs/FeatSharp) |
| ![Star](https://img.shields.io/github/stars/icandle/GenDR.svg?style=social&label=Star) <br> [**GenDR: Lightning Generative Detail Restorator**](https://arxiv.org/abs/2503.06790) <br> | arXiv | 2025-03 | [Github](https://github.com/icandle/GenDR) |
| ![Star](https://img.shields.io/github/stars/Roveer/Patch-Based-Adapter.svg?style=social&label=Star) <br> [**Ultra High-Resolution Image Inpainting with Patch-Based Content Consistency Adapter**](https://arxiv.org/abs/2510.13419) <br> | arXiv | 2025-10 | [Github](https://github.com/Roveer/Patch-Based-Adapter) |

##### <a id="112-feature-fusion">1.1.2 Feature Fusion</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/TempleX98/MoVA.svg?style=social&label=Star) <br> [**MoVA: Adapting Mixture of Vision Experts to Multimodal Context**](https://arxiv.org/abs/2404.13046) <br> | NeurIPS | 2024-04 | [Github](https://github.com/TempleX98/MoVA) |
| [**CLIP-aware Domain-Adaptive Super-Resolution**](https://link.springer.com/article/10.1007/s00530-025-01849-8) <br> | MMS | 2025-05 | [-](-) |
| ![Star](https://img.shields.io/github/stars/DragonisCV/RAM.svg?style=social&label=Star) <br> [**RAM++: Robust Representation Learning via Adaptive Mask for All-in-One Image Restoration**](https://arxiv.org/abs/2509.12039) <br> | arXiv | 2025-09 | [Github](https://github.com/DragonisCV/RAM) |
| [**Vision-Language Alignment from Compressed Image Representations using 2D Gaussian Splatting**](https://arxiv.org/abs/2509.22615) <br> | arXiv | 2025-09 | [-](-) |

#### <a id="12-language-branch-adaptation-bridging-modalities">1.2 Language Branch Adaptation: Bridging Modalities</a>

##### <a id="121-prompt-learning-strategies">1.2.1 Prompt Learning Strategies</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/Yaphabates/RESTORE_.svg?style=social&label=Star) <br> [**RESTORE: Towards Feature Shift for Vision-Language Prompt Learning**](https://arxiv.org/abs/2403.06136) <br> | arXiv | 2024-03 | [Github](https://github.com/Yaphabates/RESTORE_) |
| ![Star](https://img.shields.io/github/stars/GaoMY-521/CaPL_Code.svg?style=social&label=Star) <br> [**Causality-guided Prompt Learning for Vision-language Models via Visual Granulation**](https://arxiv.org/abs/2509.03803) <br> | ICCV | 2025-09 | [Github](https://github.com/GaoMY-521/CaPL_Code) |

##### <a id="122-instruction-tuning-for-restoration">1.2.2 Instruction Tuning for Restoration</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**Lumina-OmniLV: A Unified Multimodal Framework for General Low-Level Vision**](https://arxiv.org/abs/2504.04903) <br> | arXiv | 2025-04 | [-](-) |

#### <a id="13-output-head-adaptation-from-tokens-to-pixels">1.3 Output Head Adaptation: From Tokens to Pixels</a>

##### <a id="131-tokenizerdecoder-framework">1.3.1 Tokenizer–Decoder Framework</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/zh460045050/V2L-Tokenizer.svg?style=social&label=Star) <br> [**Beyond Text: Frozen Large Language Models in Visual Signal Comprehension**](https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_Beyond_Text_Frozen_Large_Language_Models_in_Visual_Signal_Comprehension_CVPR_2024_paper.html) <br> | CVPR | 2024-03 | [Github](https://github.com/zh460045050/V2L-Tokenizer) |
| ![Star](https://img.shields.io/github/stars/nonwhy/PURE.svg?style=social&label=Star) <br> [**Perceive, Understand and Restore: Real-World Image Super-Resolution with Autoregressive Multimodal Generative Models**](https://openaccess.thecvf.com/content/ICCV2025/html/Wei_Perceive_Understand_and_Restore_Real-World_Image_Super-Resolution_with_Autoregressive_Multimodal_ICCV_2025_paper.html) <br> | ICCV | 2025-03 | [Github](https://github.com/nonwhy/PURE) |
| [**SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook for Multimodal Understanding and Generation**](http://arxiv.org/abs/2503.06764) <br> | arXiv | 2025-03 | [-](-) |
| ![Star](https://img.shields.io/github/stars/Joyies/TVT.svg?style=social&label=Star) <br> [**Fine-structure Preserved Real-world Image Super-resolution via Transfer VAE Training**](https://arxiv.org/abs/2507.20291) <br> | ICCV | 2025-07 | [Github](https://github.com/Joyies/TVT) |

#### <a id="14-parameter-efficient-fine-tuning-in-restoration">1.4 Parameter-Efficient Fine-Tuning in Restoration</a>

##### <a id="141-lora--adapter-integration">1.4.1 LoRA & Adapter Integration</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/cswry/OSEDiff.svg?style=social&label=Star) <br> [**One-Step Effective Diffusion Network for Real-World Image Super-Resolution**](https://arxiv.org/abs/2406.08177) <br> | NeurIPS | 2024-06 | [Github](https://github.com/cswry/OSEDiff) |
| [**Acquire and then Adapt: Squeezing out Text-to-Image Model for Image Restoration**](http://arxiv.org/abs/2504.15159) <br> | CVPR | 2025-04 | [-](-) |
| [**Demystifying the Visual Quality Paradox in Multimodal Large Language Models**](https://arxiv.org/abs/2506.15645) <br> | arXiv | 2025-06 | [-](-) |
| [**RDDM: Practicing RAW Domain Diffusion Model for Real-world Image Restoration**](https://arxiv.org/abs/2508.19154) <br> | arXiv | 2025-08 | [-](-) |
| [**BIR-Adapter: A Low-Complexity Diffusion Model Adapter for Blind Image Restoration**](https://arxiv.org/abs/2509.06904) <br> | arXiv | 2025-09 | [-](-) |
| ![Star](https://img.shields.io/github/stars/RedMediaTech/ODTSR.svg?style=social&label=Star) <br> [**One-Step Diffusion Transformer for Controllable Real-World Image Super-Resolution**](https://arxiv.org/abs/2511.17138) <br> | arXiv | 2025-11 | [Github](https://github.com/RedMediaTech/ODTSR) |

##### <a id="142-freezing-strategies">1.4.2 Freezing Strategies</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/bytetriper/LM4LV.svg?style=social&label=Star) <br> [**LM4LV: A Frozen Large Language Model for Low-level Vision Tasks**](https://arxiv.org/abs/2405.15734) <br> | arXiv | 2024-05 | [Github](https://github.com/bytetriper/LM4LV) |
| ![Star](https://img.shields.io/github/stars/ziqipang/LM4VisualEncoding.svg?style=social&label=Star) <br> [**Frozen Transformers in Language Models Are Effective Visual Encoder Layers**](https://openreview.net/forum?id=t0FI3Q66K5) <br> | ICLR | 2023-10 | [Github](https://github.com/ziqipang/LM4VisualEncoding) |

---

### <a id="2-vlm-as-auxiliary-for-low-level-vision">2. VLM as Auxiliary for Low-Level Vision</a>

#### <a id="21-vlm-as-semantic-provider-text-guided-restoration">2.1 VLM as Semantic Provider: Text-Guided Restoration</a>

##### <a id="211-text-conditioned-injection">2.1.1 Text-Conditioned Injection</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/cswry/SeeSR.svg?style=social&label=Star) <br> [**SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution**](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_SeeSR_Towards_Semantics-Aware_Real-World_Image_Super-Resolution_CVPR_2024_paper.html) <br> | CVPR | 2023-11 | [Github](https://github.com/cswry/SeeSR) |
| ![Star](https://img.shields.io/github/stars/hejh8/CFWD.svg?style=social&label=Star) <br> [**Low-light Image Enhancement via CLIP-Fourier Guided Wavelet Diffusion**](https://arxiv.org/abs/2401.03788) <br> | TOMM | 2024-01 | [Github](https://github.com/hejh8/CFWD) |
| ![Star](https://img.shields.io/github/stars/Zhaozixiang1228/IF-FILM.svg?style=social&label=Star) <br> [**Image Fusion via Vision-Language Model**](https://arxiv.org/abs/2402.02235) <br> | ICML | 2024-02 | [Github](https://github.com/Zhaozixiang1228/IF-FILM) |
| ![Star](https://img.shields.io/github/stars/xiaogang00/pretrain_model_boost_restoration.svg?style=social&label=Star) <br> [**Boosting Image Restoration via Priors from Pre-trained Models**](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Boosting_Image_Restoration_via_Priors_from_Pre-trained_Models_CVPR_2024_paper.html) <br> | CVPR | 2024-03 | [Github](https://github.com/xiaogang00/pretrain_model_boost_restoration) |
| ![Star](https://img.shields.io/github/stars/skipper-zc/MMGInpainting.svg?style=social&label=Star) <br> [**MMGInpainting: Multi-Modality Guided Image Inpainting Based on Diffusion Models**](https://doi.org/10.1109/TMM.2024.3382484) <br> | TMM | 2024-03 | [Github](https://github.com/skipper-zc/MMGInpainting) |
| ![Star](https://img.shields.io/github/stars/XunpengYi/Text-IF.svg?style=social&label=Star) <br> [**Text-IF: Leveraging Semantic Text Guidance for Degradation-Aware and Interactive Image Fusion**](https://openaccess.thecvf.com/content/CVPR2024/html/Yi_Text-IF_Leveraging_Semantic_Text_Guidance_for_Degradation-Aware_and_Interactive_Image_CVPR_2024_paper.html) <br> | CVPR | 2024-03 | [Github](https://github.com/XunpengYi/Text-IF) |
| ![Star](https://img.shields.io/github/stars/qyp2000/XPSR.svg?style=social&label=Star) <br> [**XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution**](https://arxiv.org/abs/2403.05049) <br> | ECCV | 2024-03 | [Github](https://github.com/qyp2000/XPSR) |
| ![Star](https://img.shields.io/github/stars/Algolzw/daclip-uir.svg?style=social&label=Star) <br> [**Photo-Realistic Image Restoration in the Wild with Controlled Vision-Language Models**](https://arxiv.org/abs/2404.09732) <br> | CVPRW | 2024-04 | [Github](https://github.com/Algolzw/daclip-uir) |
| [**Unsupervised Image Prior via Prompt Learning and CLIP Semantic Guidance for Low-Light Image Enhancement**](https://arxiv.org/abs/2405.11478) <br> | CVPRW | 2024-05 | [-](-) |
| ![Star](https://img.shields.io/github/stars/zyhrainbow/SSP-IR.svg?style=social&label=Star) <br> [**SSP-IR: Semantic and Structure Priors for Diffusion-based Realistic Image Restoration**](https://arxiv.org/abs/2407.03635) <br> | TCSVT | 2024-07 | [Github](https://github.com/zyhrainbow/SSP-IR) |
| ![Star](https://img.shields.io/github/stars/zhaolb4080/MTG-Fusion.svg?style=social&label=Star) <br> [**Multi-Text Guidance Is Important: Multi-Modality Image Fusion via Large Generative Vision-Language Model**](https://doi.org/10.1007/s11263-025-02409-3) <br> | IJCV | 2025-02 | [Github](https://github.com/zhaolb4080/MTG-Fusion) |
| ![Star](https://img.shields.io/github/stars/striveAgain/MegaSR.svg?style=social&label=Star) <br> [**MegaSR: Mining Customized Semantics and Expressive Guidance for Image Super-Resolution**](https://arxiv.org/abs/2503.08096) <br> | arXiv | 2025-03 | [Github](https://github.com/striveAgain/MegaSR) |
| [**The Power of Context: How Multimodality Improves Image Super-Resolution**](https://arxiv.org/abs/2503.14503) <br> | CVPR | 2025-03 | [-](-) |
| [**Coarse-to-fine text injecting for realistic image super-resolution**](https://doi.org/10.1016/j.neucom.2025.129591) <br> | Neurocom | 2025-02 | [-](-) |
| ![Star](https://img.shields.io/github/stars/Wenyuzhy/DeepSPG.svg?style=social&label=Star) <br> [**DeepSPG: Exploring Deep Semantic Prior Guidance for Low-light Image Enhancement with Multimodal Learning**](https://arxiv.org/abs/2504.19127) <br> | arXiv | 2025-04 | [Github](https://github.com/Wenyuzhy/DeepSPG) |
| [**CTD-inpainting: Towards the Coherence of Text-driven Inpainting with Blended Diffusion**](https://doi.org/10.1016/j.inffus.2025.103163) <br> | Information Fusion | 2025-04 | [-](-) |
| ![Star](https://img.shields.io/github/stars/ywxjm/Diff-Dehazer.svg?style=social&label=Star) <br> [**Exploiting Diffusion Prior for Real-World Image Dehazing with Unpaired Training**](https://arxiv.org/abs/2503.15017) <br> | AAAI | 2025-03 | [Github](https://github.com/ywxjm/Diff-Dehazer) |
| ![Star](https://img.shields.io/github/stars/noxsine/GPT_Restoration.svg?style=social&label=Star) <br> [**A Preliminary Study for GPT-4o on Image Restoration**](https://arxiv.org/abs/2505.05621) <br> | arXiv | 2025-05 | [Github](https://github.com/noxsine/GPT_Restoration) |
| ![Star](https://img.shields.io/github/stars/bryanswkim/Chain-of-Zoom.svg?style=social&label=Star) <br> [**Chain-of-Zoom: Extreme Super-Resolution via Scale Autoregression and Preference Alignment**](https://arxiv.org/abs/2505.18600) <br> | NeurIPS  | 2025-05 | [Github](https://github.com/bryanswkim/Chain-of-Zoom) |
| ![Star](https://img.shields.io/github/stars/cvlab-kaist/TAIR.svg?style=social&label=Star) <br> [**Text-Aware Image Restoration with Diffusion Models**](https://arxiv.org/abs/2506.09993) <br> | arXiv | 2025-06 | [Github](https://github.com/cvlab-kaist/TAIR) |
| ![Star](https://img.shields.io/github/stars/AMAP-ML/LD-RPS.svg?style=social&label=Star) <br> [**LD-RPS: Zero-Shot Unified Image Restoration via Latent Diffusion Recurrent Posterior Sampling**](https://arxiv.org/abs/2507.00790) <br> | ICCV | 2025-07 | [Github](https://github.com/AMAP-ML/LD-RPS) |
| [**LLaVA-based semantic feature modulation diffusion model for underwater image enhancement**](https://doi.org/10.1016/j.inffus.2025.103566) <br> Category: **2.1.1 Text-Conditioned Injection** <br> | InfFus | 2025-07 | [-](-) |
| ![Star](https://img.shields.io/github/stars/sunxiaoran01/VLM-IMI.svg?style=social&label=Star) <br> [**Adapting Large VLMs with Iterative and Manual Instructions for Generative Low-light Enhancement**](https://arxiv.org/abs/2507.18064) <br> | arXiv | 2025-07 | [Github](https://github.com/sunxiaoran01/VLM-IMI) |
| ![Star](https://img.shields.io/github/stars/Fediory/HVI-CIDNet.svg?style=social&label=Star) <br> [**HVI-CIDNet+: Beyond Extreme Darkness for Low-Light Image Enhancement**](https://arxiv.org/abs/2507.06814) <br> | arXiv | 2025-07 | [Github](https://github.com/Fediory/HVI-CIDNet) |
| ![Star](https://img.shields.io/github/stars/albrateanu/ModalFormer.svg?style=social&label=Star) <br> [**ModalFormer: Multimodal Transformer for Low-Light Image Enhancement**](https://arxiv.org/abs/2507.20388) <br> | arXiv | 2025-07 | [Github](https://github.com/albrateanu/ModalFormer) |
| ![Star](https://img.shields.io/github/stars/Feecuin/AWM-Fuse.svg?style=social&label=Star) <br> [**AWM-Fuse: Multi-Modality Image Fusion for Adverse Weather via Global and Local Text Perception**](https://arxiv.org/abs/2508.16881) <br> | arXiv | 2025-08 | [Github](https://github.com/Feecuin/AWM-Fuse) |
| [**MambaTrans: Multimodal Fusion Image Translation via Large Language Model Priors for Downstream Visual Tasks**](https://arxiv.org/abs/2508.07803) <br> | arXiv | 2025-08 | [-](-) |
| ![Star](https://img.shields.io/github/stars/W2GenAI-Lab/LucidFlux.svg?style=social&label=Star) <br> [**LucidFlux: Caption-Free Universal Image Restoration via a Large-Scale Diffusion Transformer**](https://arxiv.org/abs/2509.22414) <br> | arXiv | 2025-09 | [Github](https://github.com/W2GenAI-Lab/LucidFlux) |
| [**Vision-Language Model Guided Image Restoration**](https://arxiv.org/abs/2512.17292) <br> | arXiv | 2025-12 | [-](-) |
| [**Zero-Shot Image Super-Resolution Using Prompt-Driven Vision-Language Foundation Models Without Task-Specific Fine-Tuning**](https://doi.org/10.21203/rs.3.rs-7346896/v1) <br> | arXiv | 2025-09 | [-](-) |
| [**Extreme Blind Image Restoration via Prompt-Conditioned Information Bottleneck**](https://arxiv.org/abs/2510.00728) <br> | arXiv | 2025-10 | [-](-) |
| ![Star](https://img.shields.io/github/stars/liuyunjing0306/CINet.svg?style=social&label=Star) <br> [**A structural information-guided cross-modal method for damaged inscription inpainting via vision-language models**](https://doi.org/10.1038/s40494-025-02059-1) <br> | npj Heritage Science | 2025-09 | [Github](https://github.com/liuyunjing0306/CINet) |
| [**GLYPH-SR: Can We Achieve Both High-Quality Image Super-Resolution and High-Fidelity Text Recovery via VLM-Guided Latent Diffusion Model?**](https://arxiv.org/abs/2510.26339) <br> | arXiv | 2025-10 | [-](-) |
| ![Star](https://img.shields.io/github/stars/cvlab-kaist/UniT.svg?style=social&label=Star) <br> [**Unified Diffusion Transformer for High-fidelity Text-Aware Image Restoration**](https://arxiv.org/abs/2512.08922) <br> | arXiv | 2025-12 | [Github](https://github.com/cvlab-kaist/UniT) |

##### <a id="212-language-driven-manipulation">2.1.2 Language-Driven Manipulation</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/mv-lab/InstructIR.svg?style=social&label=Star) <br> [**InstructIR: High-Quality Image Restoration Following Human Instructions**](https://arxiv.org/abs/2401.16468) <br> | ECCV | 2024-01 | [Github](https://github.com/mv-lab/InstructIR) |
| ![Star](https://img.shields.io/github/stars/KVGandikota/Text-guidedSR.svg?style=social&label=Star) <br> [**Text-guided Explorable Image Super-resolution**](https://arxiv.org/abs/2403.01124) <br> | CVPR | 2024-03 | [Github](https://github.com/KVGandikota/Text-guidedSR) |
| ![Star](https://img.shields.io/github/stars/RotsteinNoam/Paint-by-Inpaint.svg?style=social&label=Star) <br> [**Paint by Inpaint: Learning to Add Image Objects by Removing Them First**](https://arxiv.org/abs/2404.18212) <br> | CVPR | 2024-04 | [Github](https://github.com/RotsteinNoam/Paint-by-Inpaint) |
| [**Brush2Prompt: Contextual Prompt Generator for Object Inpainting**](https://openaccess.thecvf.com/content/CVPR2024/html/Chiu_Brush2Prompt_Contextual_Prompt_Generator_for_Object_Inpainting_CVPR_2024_paper.html) <br> | CVPR | 2024-06 | [-](-) |
| ![Star](https://img.shields.io/github/stars/SherryXTChen/Instruct-CLIP.svg?style=social&label=Star) <br> [**Instruct-CLIP: Improving Instruction-Guided Image Editing with Automated Data Refinement Using Contrastive Learning**](https://arxiv.org/abs/2503.18406) <br> | CVPR | 2025-03 | [Github](https://github.com/SherryXTChen/Instruct-CLIP) |
| [**Instruct2See: Learning to Remove Any Obstructions Across Distributions**](https://arxiv.org/abs/2505.17649) <br> | ICML | 2025-05 | [Project](https://jhscut.github.io/Instruct2See/) |
| [**MIND-Edit: MLLM Insight-Driven Editing via Language-Vision Projection**](https://arxiv.org/abs/2505.19149) <br> | arXiv | 2025-05 | [-](-) |
| ![Star](https://img.shields.io/github/stars/AdvancedPhotonSource/aether.svg?style=social&label=Star) <br> [**Fidelity-Preserving Enhancement of Ptychography with Foundational Text-to-Image Models**](https://arxiv.org/abs/2509.04513) <br> | arXiv | 2025-09 | [Github](https://github.com/AdvancedPhotonSource/aether) |
| [**FOCUS: Unified Vision-Language Modeling for Interactive Editing Driven by Referential Segmentation**](https://arxiv.org/abs/2506.16806) <br> | NeurIPS | 2025-06 | [-](-) |
| [**LaTo: Landmark-Tokenized Diffusion Transformer for Fine-Grained Human Face Editing**](https://arxiv.org/abs/2509.25731) <br> | arXiv | 2025-09 | [-](-) |
| [**Query-Kontext: An Unified Multimodal Model for Image Generation and Editing**](https://arxiv.org/abs/2509.26641) <br> | arXiv | 2025-09 | [-](-) |
| ![Star](https://img.shields.io/github/stars/HiDream-ai/VAREdit.svg?style=social&label=Star) <br> [**Visual Autoregressive Modeling for Instruction-Guided Image Editing**](https://arxiv.org/abs/2508.15772) <br> | arXiv | 2025-08 | [Github](https://github.com/HiDream-ai/VAREdit) |
| ![Star](https://img.shields.io/github/stars/snap-research/kontinuouskontext.svg?style=social&label=Star) <br> [**Kontinuous Kontext: Continuous Strength Control for Instruction-based Image Editing**](https://arxiv.org/abs/2510.08532) <br> | arXiv | 2025-10 | [Github](https://github.com/snap-research/kontinuouskontext) |
| ![Star](https://img.shields.io/github/stars/thuyvuphuong/Region-in-Context.svg?style=social&label=Star) <br> [**Region in Context: Text-Conditioned Image Editing with Human-Like Semantic Reasoning**](https://arxiv.org/abs/2510.16772) <br> | arXiv | 2025-10 | [Github](https://github.com/thuyvuphuong/Region-in-Context) |
| [**UniSER: A Foundation Model for Unified Soft Effects Removal**](https://arxiv.org/abs/2511.14183) <br> | arXiv | 2025-11 | [-](-) |
| [**Generative Editing in the Joint Vision-Language Space for Zero-Shot Composed Image Retrieval**](https://arxiv.org/abs/2512.01636) <br> | arXiv | 2025-12 | [-](-) |

##### <a id="213-subject-aware-restoration">2.1.3 Subject-Aware Restoration</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/shuaizhengliu/InstructRestore.svg?style=social&label=Star) <br> [**InstructRestore: Region-Customized Image Restoration with Human Instructions**](https://arxiv.org/abs/2503.24357) <br> | arXiv | 2025-03 | [Github](https://github.com/shuaizhengliu/InstructRestore) |
| [**PromptLNet: Region-Adaptive Aesthetic Enhancement via Prompt Guidance in Low-Light Enhancement Net**](https://arxiv.org/abs/2503.08276) <br> | arXiv | 2025-03 | [-](-) |
| [**TSCnet: A Text-driven Semantic-level Controllable Framework for Customized Low-Light Image Enhancement**](http://arxiv.org/abs/2503.08168) <br> | Neurocomputing | 2025-03 | [Project](https://miaorain.github.io/lowlight09.github.io/) |
| [**EAM: Enhancing Anything with Diffusion Transformers for Blind Super-Resolution**](https://arxiv.org/abs/2505.05209) <br> | arXiv | 2025-05 | [-](-) |

#### <a id="22-vlm-as-degradation-interpreter-visual-prompting--context">2.2 VLM as Degradation Interpreter: Visual-Prompting & Context</a>

##### <a id="221-degradation-classification">2.2.1 Degradation Classification</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/puppy210/DaLPSR.svg?style=social&label=Star) <br> [**DaLPSR: Leverage Degradation-Aligned Language Prompt for Real-World Image Super-Resolution**](https://arxiv.org/abs/2406.16477) <br> | arXiv | 2024-06 | [Github](https://github.com/puppy210/DaLPSR) |
| [**DAP-LED: Learning Degradation-Aware Priors with CLIP for Joint Low-light Enhancement and Deblurring**](http://arxiv.org/abs/2409.13496) <br> | ICRA | 2024-09 | [-](-) |
| [**Multi-modal degradation feature learning for unified image restoration based on contrastive learning**](https://doi.org/10.1016/j.neucom.2024.128955) <br> | Neurocomputing | 2024-11 | [-](-) |
| ![Star](https://img.shields.io/github/stars/xianggkl/VLU-Net.svg?style=social&label=Star) <br> [**Vision-Language Gradient Descent-driven All-in-One Deep Unfolding Networks**](https://arxiv.org/abs/2503.16930) <br> | CVPR | 2025-03 | [Github](https://github.com/xianggkl/VLU-Net) |
| [**DA2Diff: Exploring Degradation-aware Adaptive Diffusion Priors for All-in-One Weather Restoration**](https://arxiv.org/abs/2504.05135) <br> | arXiv | 2025-04 | [-](-) |
| [**VL-UR: Vision-Language-guided Universal Restoration of Images Degraded by Adverse Weather Conditions**](https://arxiv.org/abs/2504.08219) <br> | ICME | 2025-04 | [-](-) |
| [**Controlling vision-language model for enhancing image restoration**](https://doi.org/10.1016/j.imavis.2025.105538) <br> | IVC | 2025-04 | [-](-) |
| [**CLIP-driven rain perception: Adaptive deraining with pattern-aware network routing and mask-guided cross-attention**](https://www.sciencedirect.com/science/article/pii/S0031320325015493) <br> | PR | 2025-06 | [-](-) |
| [**Degradation-Aware Image Enhancement via Vision-Language Classification**](https://arxiv.org/abs/2506.05450) <br> | arXiv | 2025-06 | [-](-) |
| ![Star](https://img.shields.io/github/stars/yz-wang/M2Restore.svg?style=social&label=Star) <br> [**M2Restore: Mixture-of-Experts-based Mamba-CNN Fusion Framework for All-in-One Image Restoration**](https://arxiv.org/abs/2506.07814) <br> | TIP | 2025-06 | [Github](https://github.com/yz-wang/M2Restore) |
| [**Real-world super-resolution with VLM-based degradation prior learning**](https://www.nature.com/articles/s41598-025-14581-0) <br> | Sci Rep | 2025-08 | [-](-) |
| [**Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution**](https://arxiv.org/abs/2511.16024) <br> | arXiv | 2025-11 | [-](-) |
| ![Star](https://img.shields.io/github/stars/LowLevelAI/VAR-LIDE.svg?style=social&label=Star) <br> [**Zero-Reference Joint Low-Light Enhancement and Deblurring via Visual Autoregressive Modeling with VLM-Derived Modulation**](https://arxiv.org/abs/2511.18591) <br> | AAAI | 2025-11 | [Github](https://github.com/LowLevelAI/VAR-LIDE) |

##### <a id="222-description-based-restoration">2.2.2 Description-based Restoration</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**LLMRA: Multi-modal Large Language Model based Restoration Assistant**](https://arxiv.org/abs/2401.11401) <br> | arXiv | 2024-01 | [-](-) |
| ![Star](https://img.shields.io/github/stars/zyhrainbow/Diff-Restorer.svg?style=social&label=Star) <br> [**Diff-Restorer: Unleashing Visual Prompts for Diffusion-based Universal Image Restoration**](https://arxiv.org/abs/2407.03636) <br> | arXiv | 2024-07 | [Github](https://github.com/zyhrainbow/Diff-Restorer) |
| [**Training-Free Large Model Priors for Multiple-in-One Image Restoration**](https://arxiv.org/abs/2407.13181) <br> | arXiv | 2024-07 | [-](-) |
| ![Star](https://img.shields.io/github/stars/IntMeGroup/UniProcessor.svg?style=social&label=Star) <br> [**UniProcessor: A Text-induced Unified Low-level Image Processor**](https://arxiv.org/abs/2407.20928) <br> | ECCV | 2024-07 | [Github](https://github.com/IntMeGroup/UniProcessor) |
| ![Star](https://img.shields.io/github/stars/sudraj2002/AWRaCLe.svg?style=social&label=Star) <br> [**AWRaCLe: All-Weather Image Restoration using Visual In-Context Learning**](https://arxiv.org/abs/2409.00263) <br> | AAAI | 2024-08 | [Github](https://github.com/sudraj2002/AWRaCLe) |
| [**Leveraging vision-language prompts for real-world image restoration and enhancement**](https://www.sciencedirect.com/science/article/abs/pii/S1077314224003035) <br> | CVIU | 2024-11 | [-](-) |
| ![Star](https://img.shields.io/github/stars/igor-morawski/tmm-sem.svg?style=social&label=Star) <br> [**Leveraging Content and Context Cues for Low-Light Image Enhancement**](https://arxiv.org/abs/2412.07693) <br> | TMM | 2024-12 | [Github](https://github.com/igor-morawski/tmm-sem) |
| [**RamIR: Reasoning and action prompting with Mamba for all-in-one image restoration**](https://link.springer.com/article/10.1007/s10489-024-06226-y) <br> | - | 2025-01 | [-](-) |
| ![Star](https://img.shields.io/github/stars/Linfeng-Tang/ControlFusion.svg?style=social&label=Star) <br> [**ControlFusion: A Controllable Image Fusion Network with Language-Vision Degradation Prompts**](https://arxiv.org/abs/2503.23356) <br> | arXiv | 2025-03 | [Github](https://github.com/Linfeng-Tang/ControlFusion) |
| ![Star](https://img.shields.io/github/stars/RongxinL/CyclicPrompt.svg?style=social&label=Star) <br> [**Prompt to Restore, Restore to Prompt: Cyclic Prompting for Universal Adverse Weather Removal**](https://arxiv.org/abs/2503.09013) <br> | TIP | 2025-03 | [Github](https://github.com/RongxinL/CyclicPrompt) |
| [**UniCoRN: Latent Diffusion-based Unified Controllable Image Restoration Network across Multiple Degradations**](https://arxiv.org/abs/2503.15868) <br> | WACV | 2025-03 | [Project](https://codejaeger.github.io/unicorn-gh/) |
| ![Star](https://img.shields.io/github/stars/HXDreamChaser/CLIP-RestoreX.svg?style=social&label=Star) <br> [**CLIP-RestoreX: Restore Image Structure and Perception in Exposure Correction**](https://ojs.aaai.org/index.php/AAAI/article/view/32392) <br> | AAAI | 2025-04 | [Github](https://github.com/HXDreamChaser/CLIP-RestoreX) |
| ![Star](https://img.shields.io/github/stars/kongdehong/DPIR.svg?style=social&label=Star) <br> [**Dual Prompting Image Restoration with Diffusion Transformers**](https://openaccess.thecvf.com/content/CVPR2025/html/Kong_Dual_Prompting_Image_Restoration_with_Diffusion_Transformers_CVPR_2025_paper.html) <br> | CVPR | 2025-04 | [Github](https://github.com/kongdehong/DPIR) |
| ![Star](https://img.shields.io/github/stars/CV-Rookie/A-single-image-deraining-algorithm-guided-by-text-generation-based-on-depth-information-conditions.svg?style=social&label=Star) <br> [**A single image deraining algorithm guided by text generation based on depth information conditions**](https://doi.org/10.1016/j.asoc.2025.113506) <br> | ASOC | 2025-07 | [Github](https://github.com/CV-Rookie/A-single-image-deraining-algorithm-guided-by-text-generation-based-on-depth-information-conditions) |
| ![Star](https://img.shields.io/github/stars/buuzhangsen/AM-PromptIR.svg?style=social&label=Star) <br> [**Adaptive multi-modal prompting for universal image restoration amidst diverse degradations**](https://doi.org/10.1007/s00371-025-04023-3) <br> | TVC | 2025-05 | [Github](https://github.com/buuzhangsen/AM-PromptIR) |
| [**RAGSR: Regional Attention Guided Diffusion for Image Super-Resolution**](https://arxiv.org/abs/2508.16158) <br> | arXiv | 2025-08 | [-](-) |
| [**Dual-Domain Perspective on Degradation-Aware Fusion: A VLM-Guided Robust Infrared and Visible Image Fusion Framework**](https://arxiv.org/abs/2509.05000) <br> | arXiv | 2025-09 | [-](-) |
| [**Real-world super-resolution with VLM-based degradation prior learning**](https://www.nature.com/articles/s41598-025-14581-0) <br> | Scientific Reports | 2025-08 | [-](-) |
| ![Star](https://img.shields.io/github/stars/Lmmh058/VGDCFusion.svg?style=social&label=Star) <br> [**Coupled Degradation Modeling and Fusion: A VLM-Guided Degradation-Coupled Network for Degradation-Aware Infrared and Visible Image Fusion**](https://arxiv.org/abs/2510.11456) <br> | arXiv | 2025-10 | [Github](https://github.com/Lmmh058/VGDCFusion) |
| ![Star](https://img.shields.io/github/stars/Lmmh058/VGDCFusion.svg?style=social&label=Star) <br> [**Coupled Degradation Modeling and Fusion: A VLM-Guided Degradation-Coupled Network for Degradation-Aware Infrared and Visible Image Fusion**](https://arxiv.org/abs/2510.11456) <br> | arXiv | 2025-10 | [Github](https://github.com/Lmmh058/VGDCFusion) |
| ![Star](https://img.shields.io/github/stars/doudou845133/MdaIF.svg?style=social&label=Star) <br> [**MdaIF: Robust One-Stop Multi-Degradation-Aware Image Fusion with Language-Driven Semantics**](https://arxiv.org/abs/2511.12525) <br> | AAAI | 2025-11 | [Github](https://github.com/doudou845133/MdaIF) |
| ![Star](https://img.shields.io/github/stars/Young-spec-design/Image-Fusion-Network.svg?style=social&label=Star) <br> [**Multimodal image fusion network with prior-guided dynamic degradation removal for extreme environment perception**](https://doi.org/10.1038/s41598-025-24436-3) <br> | Sci Rep | 2025-10 | [Github](https://github.com/Young-spec-design/Image-Fusion-Network) |
| [**Nested Unfolding Network for Real-World Concealed Object Segmentation**](https://arxiv.org/abs/2511.18164) <br> | arXiv | 2025-11 | [-](-) |
| [**VL-UR: Vision-Language-guided Universal Restoration of Images Degraded by Adverse Weather Conditions**](https://arxiv.org/abs/2504.08219) <br> | arXiv | 2025-04 | [-](-) |
| [**VLM-Augmented Degradation Modeling for Image Restoration Under Adverse Weather Conditions**](https://arxiv.org/abs/2511.16998) <br> | arXiv | 2025-11 | [-](-) |

#### <a id="23-vlm-as-quality-evaluator-perception-and-assessment">2.3 VLM as Quality Evaluator: Perception and Assessment</a>

##### <a id="231-no-reference-quality-assessment">2.3.1 No-Reference Quality Assessment</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/miccunifi/QualiCLIP.svg?style=social&label=Star) <br> [**Quality-Aware Image-Text Alignment for Opinion-Unaware Image Quality Assessment (QualiCLIP)**](https://arxiv.org/abs/2403.11176) <br> | arXiv | 2024-03 | [Github](https://github.com/miccunifi/QualiCLIP) |
| ![Star](https://img.shields.io/github/stars/XPixelGroup/DepictQA.svg?style=social&label=Star) <br> [**Descriptive Image Quality Assessment in the Wild**](https://arxiv.org/abs/2405.18842) <br> | arXiv | 2024-05 | [Github](https://github.com/XPixelGroup/DepictQA) |
| ![Star](https://img.shields.io/github/stars/zht8506/UniQA.svg?style=social&label=Star) <br> [**UniQA: Unified Vision-Language Pre-training for Image Quality and Aesthetic Assessment**](https://arxiv.org/abs/2406.01069) <br> | arXiv | 2024-06 | [Github](https://github.com/zht8506/UniQA) |
| ![Star](https://img.shields.io/github/stars/wzczc/CLIP-AGIQA.svg?style=social&label=Star) <br> [**CLIP-AGIQA: Boosting the Performance of AI-Generated Image Quality Assessment with CLIP**](https://arxiv.org/abs/2408.15098) <br> | ICPR | 2024-08 | [Github](https://github.com/wzczc/CLIP-AGIQA) |
| ![Star](https://img.shields.io/github/stars/LowLevelAI/GPP-LLIE.svg?style=social&label=Star) <br> [**Low-Light Image Enhancement via Generative Perceptual Priors**](https://arxiv.org/abs/2412.20916) <br> | AAAI | 2024-12 | [Github](https://github.com/LowLevelAI/GPP-LLIE) |
| ![Star](https://img.shields.io/github/stars/zhiyuanyou/DeQA-Score.svg?style=social&label=Star) <br> [**Teaching Large Language Models to Regress Accurate Image Quality Scores using Score Distribution**](https://arxiv.org/abs/2501.11561) <br> | CVPR | 2025-01 | [Github](https://github.com/zhiyuanyou/DeQA-Score) |
| ![Star](https://img.shields.io/github/stars/JunFu1995/CLIP-DQA.svg?style=social&label=Star) <br> [**CLIP-DQA: Blindly Evaluating Dehazed Images from Global and Local Perspectives Using CLIP**](https://arxiv.org/abs/2502.01707) <br> | ISCAS | 2025-02 | [Github](https://github.com/JunFu1995/CLIP-DQA) |
| [**DVLTA-VQA: Decoupled Vision-Language Modeling with Text-Guided Adaptation for Blind Video Quality Assessment**](https://arxiv.org/abs/2504.11733) <br> | arXiv | 2025-04 | [-](-) |
| ![Star](https://img.shields.io/github/stars/yeppp27/Q-Adapt.svg?style=social&label=Star) <br> [**Q-Adapt: Adapting LMM for Visual Quality Assessment with Progressive Instruction Tuning**](https://arxiv.org/abs/2504.01655) <br> | arXiv | 2025-04 | [Github](https://github.com/yeppp27/Q-Adapt) |
| ![Star](https://img.shields.io/github/stars/House-yuyu/Perceive-IR.svg?style=social&label=Star) <br> [**Perceive-IR: Learning to Perceive Degradation Better for All-in-One Image Restoration**](https://arxiv.org/abs/2408.15994) <br> | TIP | 2024-08 | [Github](https://github.com/House-yuyu/Perceive-IR) |
| ![Star](https://img.shields.io/github/stars/apple/ml-gie-bench.svg?style=social&label=Star) <br> [**GIE-Bench: Towards Grounded Evaluation for Text-Guided Image Editing**](https://arxiv.org/abs/2505.11493) <br> | arXiv | 2025-05 | [Github](https://github.com/apple/ml-gie-bench) |
| ![Star](https://img.shields.io/github/stars/PKU-YuanGroup/ImgEdit.svg?style=social&label=Star) <br> [**ImgEdit: A Unified Image Editing Dataset and Benchmark**](https://arxiv.org/abs/2505.20275) <br> | NeurIPS | 2025-05 | [Github](https://github.com/PKU-YuanGroup/ImgEdit) |
| ![Star](https://img.shields.io/github/stars/aimagelab/DICE.svg?style=social&label=Star) <br> [**What Changed? Detecting and Evaluating Instruction-Guided Image Edits with Multimodal Large Language Models**](https://arxiv.org/abs/2505.20405) <br> | ICCV | 2025-05 | [Github](https://github.com/aimagelab/DICE) |
| [**BPCLIP: A Bottom-up Image Quality Assessment from Distortion to Semantics Based on CLIP**](https://arxiv.org/abs/2506.17969) <br> | ICME | 2025-06 | [-](-) |
| ![Star](https://img.shields.io/github/stars/vivoCameraResearch/Q-Ponder.svg?style=social&label=Star) <br> [**Q-Ponder: A Unified Training Pipeline for Reasoning-based Visual Quality Assessment**](https://arxiv.org/abs/2506.05384) <br> | arXiv | 2025-06 | [Github](https://github.com/vivoCameraResearch/Q-Ponder) |
| [**Leveraging Vision-Language Models to Select Trustworthy Super-Resolution Samples Generated by Diffusion Models**](https://arxiv.org/abs/2506.20832) <br> | TCSVT | 2025-06 | [-](-) |
| [**VQ-Insight: Teaching VLMs for AI-Generated Video Quality Understanding via Progressive Visual Reinforcement Learning**](https://arxiv.org/abs/2506.18564) <br> | arXiv | 2025-06 | [-](-) |
| [**DehazeMamba: large multi-modal model guided single image dehazing via mamba**](https://doi.org/10.1007/s44267-025-00083-0) <br> | VisIntell | 2025-07 | [-](-) |
| [**Hallucination Score: Towards Mitigating Hallucinations in Generative Image Super-Resolution**](https://arxiv.org/abs/2507.14367) <br> | arXiv | 2025-07 | [-](-) |
| [**Q-CLIP: Unleashing the Power of Vision-Language Models for Video Quality Assessment through Unified Cross-Modal Adaptation**](https://arxiv.org/abs/2508.06092) <br> | arXiv | 2025-08 | [-](-) |
| ![Star](https://img.shields.io/github/stars/sxfly99/FGResQ.svg?style=social&label=Star) <br> [**Fine-grained Image Quality Assessment for Perceptual Image Restoration**](https://arxiv.org/abs/2508.14475) <br> | AAAI | 2025-08 | [Github](https://github.com/sxfly99/FGResQ) |
| ![Star](https://img.shields.io/github/stars/jzhws/VisualQualityAssessment-RL-Trainer.svg?style=social&label=Star) <br> [**Refine-IQA: Multi-Stage Reinforcement Finetuning for Perceptual Image Quality Assessment**](https://arxiv.org/abs/2508.03763) <br> | AAAI | 2025-08 | [Github](https://github.com/jzhws/VisualQualityAssessment-RL-Trainer) |
| [**Segmenting and Understanding: Region-aware Semantic Attention for Fine-grained Image Quality Assessment with Large Language Models**](https://arxiv.org/abs/2508.07818) <br> | arXiv | 2025-08 | [-](-) |
| [**AgenticIQA: An Agentic Framework for Adaptive and Interpretable Image Quality Assessment**](https://arxiv.org/abs/2509.26006) <br> | arXiv | 2025-09 | [-](-) |
| ![Star](https://img.shields.io/github/stars/xauat-liushipeng/D2S.svg?style=social&label=Star) <br> [**Describe-to-Score: Text-Guided Efficient Image Complexity Assessment**](https://arxiv.org/abs/2509.16609) <br> | arXiv | 2025-09 | [Github](https://github.com/xauat-liushipeng/D2S) |
| ![Star](https://img.shields.io/github/stars/yahya-ben/mplug2-vp-for-nriqa.svg?style=social&label=Star) <br> [**Parameter-Efficient Adaptation of mPLUG-Owl2 via Pixel-Level Visual Prompts for NR-IQA**](https://arxiv.org/abs/2509.03494) <br> | arXiv | 2025-09 | [Github](https://github.com/yahya-ben/mplug2-vp-for-nriqa) |
| [**Revisiting Vision–Language Foundations for No-Reference Image Quality Assessment**](https://arxiv.org/abs/2509.17374) <br> | WACV | 2025-09 | [-](-) |
| ![Star](https://img.shields.io/github/stars/prazmara/defog-detection-benchmark.svg?style=social&label=Star) <br> [**From Filters to VLMs: Benchmarking Defogging Methods through Object Detection and Segmentation Performance**](https://arxiv.org/abs/2510.03906) <br> | arXiv | 2025-10 | [Github](https://github.com/prazmara/defog-detection-benchmark) |
| ![Star](https://img.shields.io/github/stars/2kxx/Q-Scorer.svg?style=social&label=Star) <br> [**Revisiting MLLM Based Image Quality Assessment: Errors and Remedy**](https://arxiv.org/abs/2511.07812) <br> | AAAI | 2025-11 | [Github](https://github.com/2kxx/Q-Scorer) |

##### <a id="232-semantic-consistency-loss">2.3.2 Semantic Consistency Loss</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/ChenzhaoNju/CCLGAN.svg?style=social&label=Star) <br> [**Cycle Contrastive Adversarial Learning for Unsupervised image Deraining**](https://arxiv.org/abs/2407.11750) <br> | arXiv | 2024-07 | [Github](https://github.com/ChenzhaoNju/CCLGAN) |
| [**Underwater Diffusion Attention Network with Contrastive Language-Image Joint Learning for Underwater Image Enhancement**](https://arxiv.org/abs/2505.19895) <br> | arXiv | 2025-05 | [-](-) |
| [**Unveiling the Underwater World: CLIP Perception Model-Guided Underwater Image Enhancement**](https://doi.org/10.1016/j.patcog.2025.111395) <br> | PR | 2025-06 | [-](-) |
| [**One-Step Diffusion-based Real-World Image Super-Resolution with Visual Perception Distillation**](https://arxiv.org/abs/2506.02605) <br> | arXiv | 2025-06 | [-](-) |
| ![Star](https://img.shields.io/github/stars/wangsen99/GEFU.svg?style=social&label=Star) <br> [**From Enhancement to Understanding: Build a Generalized Bridge for Low-light Vision via Semantically Consistent Unsupervised Fine-tuning**](https://arxiv.org/abs/2507.08380) <br> | ICCV | 2025-07 | [Github](https://github.com/wangsen99/GEFU) |
| [**WeatherCycle: Unpaired Multi-Weather Restoration via Color Space Decoupled Cycle Learning**](https://arxiv.org/abs/2509.23150) <br> | arXiv | 2025-09 | [-](-) |

##### <a id="233-feedback-loops">2.3.3 Feedback Loops</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/Troivyn/HazeCLIP.svg?style=social&label=Star) <br> [**HazeCLIP: Towards Language Guided Real-World Image Dehazing**](https://arxiv.org/abs/2407.13719) <br> | ICASSP | 2024-07 | [Github](https://github.com/Troivyn/HazeCLIP) |
| ![Star](https://img.shields.io/github/stars/jiaqixuac/WResVLM.svg?style=social&label=Star) <br> [**Towards Real-World Adverse Weather Image Restoration: Enhancing Clearness and Semantics with Vision-Language Models**](https://arxiv.org/abs/2409.02101) <br> | ECCV | 2024-09 | [Github](https://github.com/jiaqixuac/WResVLM) |
| [**DSPO: Direct Semantic Preference Optimization for Real-World Image Super-Resolution**](https://arxiv.org/abs/2504.15176) <br> | arXiv | 2025-04 | [-](-) |
| ![Star](https://img.shields.io/github/stars/alexlai2860/SnowMaster.svg?style=social&label=Star) <br> [**SnowMaster: Comprehensive Real-world Image Desnowing via MLLM with Multi-Model Feedback Optimization**](https://openaccess.thecvf.com/content/CVPR2025/html/Lai_SnowMaster_Comprehensive_Real-world_Image_Desnowing_via_MLLM_with_Multi-Model_Feedback_CVPR_2025_paper.html) <br> | CVPR | 2025-06 | [Github](https://github.com/alexlai2860/SnowMaster) |
| ![Star](https://img.shields.io/github/stars/TIGER-AI-Lab/EditReward.svg?style=social&label=Star) <br> [**EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing**](https://arxiv.org/abs/2509.26346) <br> | arXiv | 2025-09 | [Github](https://github.com/TIGER-AI-Lab/EditReward) |
| [**Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking**](https://arxiv.org/abs/2509.25787) <br> | arXiv | 2025-09 | [-](-) |
| [**Learning an Image Editing Model without Image Editing Pairs**](https://arxiv.org/abs/2510.14978) <br> | arXiv | 2025-10 | [-](-) |
| ![Star](https://img.shields.io/github/stars/lbc12345/TTPO.svg?style=social&label=Star) <br> [**Test-Time Preference Optimization for Image Restoration**](https://arxiv.org/abs/2511.19169) <br> | AAAI | 2025-11 | [Github](https://github.com/lbc12345/TTPO) |
| ![Star](https://img.shields.io/github/stars/apple/ml-unigen.svg?style=social&label=Star) <br> [**UniGen-1.5: Enhancing Image Generation and Editing through Reward Unification in Reinforcement Learning**](https://arxiv.org/abs/2511.14760) <br> | arXiv | 2025-11 | [Github](https://github.com/apple/ml-unigen) |
| ![Star](https://img.shields.io/github/stars/lwq20020127/UARE.svg?style=social&label=Star) <br> [**UARE: Unified Vision-Language Model for Image Quality Assessment, Restoration, and Enhancement**](https://arxiv.org/abs/2512.06750) <br> | arXiv | 2025-12 | [Github](https://github.com/lwq20020127/UARE) |

#### <a id="24-vlm-as-intelligent-controller-agent-based-frameworks">2.4 VLM as Intelligent Controller: Agent-Based Frameworks</a>

##### <a id="241-motivation-as-intelligent-controller">2.4.1 Motivation as Intelligent Controller</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|

##### <a id="242-tool-usage--orchestration">2.4.2 Tool Usage & Orchestration</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**Incorporating Visual Experts to Resolve the Information Loss in Multimodal Large Language Models**](https://arxiv.org/abs/2401.03105) <br> | IJCAI | 2024-01 | [-](-) |
| ![Star](https://img.shields.io/github/stars/TencentARC/BrushNet.svg?style=social&label=Star) <br> [**Image Inpainting Models are Effective Tools for Instruction-guided Image Editing**](https://arxiv.org/abs/2407.13139) <br> | CVPRW | 2024-07 | [Github](https://github.com/TencentARC/BrushNet/tree/main/InstructionGuidedEditing) |
| [**RestoreAgent: Autonomous Image Restoration Agent via Multimodal Large Language Models**](https://arxiv.org/abs/2407.18035) <br> | NeurIPS | 2024-07 | [-](-) |
| [**LLMCO4MR: LLMs-Aided Neural Combinatorial Optimization for Ancient Manuscript Restoration from Fragments with Case Studies on Dunhuang**](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09585.pdf) <br> | ECCV | 2024-11 | [-](-) |
| ![Star](https://img.shields.io/github/stars/cilabuniba/i-dream-my-painting.svg?style=social&label=Star) <br> [**I Dream My Painting: Connecting MLLMs and Diffusion Models via Prompt Generation for Text-Guided Multi-Mask Inpainting**](https://arxiv.org/abs/2411.19050) <br> | WACV | 2024-11 | [Github](https://github.com/cilabuniba/i-dream-my-painting) |
| [**Hybrid Agents for Image Restoration**](https://arxiv.org/abs/2503.10120) <br> | arXiv | 2025-03 | [-](-) |
| [**Multi-Agent Image Restoration**](https://arxiv.org/abs/2503.09403) | arXiv | 2025-03 | [-](-) |
| ![Star](https://img.shields.io/github/stars/LYL1015/JarvisIR.svg?style=social&label=Star) <br> [**JarvisIR: Elevating Autonomous Driving Perception with Intelligent Image Restoration**](https://arxiv.org/abs/2504.04158) <br> | CVPR | 2025-04 | [Github](https://github.com/LYL1015/JarvisIR) |
| [**Q-Agent: Quality-Driven Chain-of-Thought Image Restoration Agent through Robust Multimodal Large Language Model**](https://arxiv.org/abs/2504.07148) <br> | arXiv | 2025-04 | [-](-) |
| ![Star](https://img.shields.io/github/stars/yuanzhengthu/LLM4UnderwaterRobot.svg?style=social&label=Star) <br> [**LEGO: LLM-enhanced genetic optimization for underwater robot image restoration**](https://doi.org/10.1016/j.patcog.2025.111782) <br> | PR | 2025-05 | [Github](https://github.com/yuanzhengthu/LLM4UnderwaterRobot) |
| [**Light as Deception: GPT-driven Natural Relighting Against Vision-Language Pre-training Models**](https://arxiv.org/abs/2505.24227) <br> | arXiv | 2025-05 | [-](-) |
| ![Star](https://img.shields.io/github/stars/niladridutt/monetGPT.svg?style=social&label=Star) <br> [**MonetGPT: Solving Puzzles Enhances MLLMs' Image Retouching Skills**](https://arxiv.org/abs/2505.06176) <br> | TOG | 2025-05 | [Github](https://github.com/niladridutt/monetGPT) |
| [**PhotoArtAgent: Intelligent Photo Retouching with Language Model-Based Artist Agents**](https://arxiv.org/abs/2505.23130) <br> | arXiv | 2025-05 | [-](-) |
| ![Star](https://img.shields.io/github/stars/LYL1015/JarvisArt.svg?style=social&label=Star) <br> [**JarvisArt: Liberating Human Artistic Creativity via an Intelligent Photo Retouching Agent**](https://arxiv.org/abs/2506.17612) <br> | NeurIPS | 2025-06 | [Github](https://github.com/LYL1015/JarvisArt) |
| ![Star](https://img.shields.io/github/stars/TianyuCodings/EdiVal.svg?style=social&label=Star) <br> [**EdiVal-Agent: An Object-Centric Framework for Automated, Scalable, Fine-Grained Evaluation of Multi-Turn Editing**](https://arxiv.org/abs/2509.13399) <br> | arXiv | 2025-09 | [Github](https://github.com/TianyuCodings/EdiVal) |
| [**InstructVTON: Optimal Auto-Masking and Natural-Language-Guided Interactive Style Control for Inpainting-Based Virtual Try-On**](https://arxiv.org/abs/2509.20524) <br> | arXiv | 2025-09 | [-](-) |
| [**Prompt-Driven Image Analysis with Multimodal Generative AI: Detection, Segmentation, Inpainting, and Interpretation**](https://arxiv.org/abs/2509.08489) <br> | arXiv | 2025-09 | [-](-) |
| ![Star](https://img.shields.io/github/stars/MediaX-SJTU/MoA-VR.svg?style=social&label=Star) <br> [**MoA-VR: A Mixture-of-Agents System Towards All-in-One Video Restoration**](https://arxiv.org/abs/2510.08508) <br> | JSTSP | 2025-10 | [Github](https://github.com/MediaX-SJTU/MoA-VR) |
| ![Star](https://img.shields.io/github/stars/HerzogFL/VisPainter.svg?style=social&label=Star) <br> [**From Pixels to Paths: A Multi-Agent Framework for Editable Scientific Illustration**](https://arxiv.org/abs/2510.27452) <br> | arXiv | 2025-10 | [Github](https://github.com/HerzogFL/VisPainter) |
| [**Image-POSER: Reflective RL for Multi-Expert Image Generation and Editing**](https://arxiv.org/abs/2511.11780) <br> | arXiv | 2025-11 | [-](-) |
| [**T2T-VICL: Unlocking the Boundaries of Cross-Task Visual In-Context Learning via Implicit Text-Driven VLMs**](https://arxiv.org/abs/2511.16107) <br> | arXiv | 2025-11 | [-](-) |
| ![Star](https://img.shields.io/github/stars/JIA-Lab-research/RePlan.svg?style=social&label=Star) <br> [**RePlan: Reasoning-guided Region Planning for Complex Instruction-based Image Editing**](https://arxiv.org/abs/2512.16864) <br> | arXiv | 2025-12 | [Github](https://github.com/JIA-Lab-research/RePlan) |

##### <a id="243-iterative-refinement">2.4.3 Iterative Refinement</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/Sealical/anywhere-multi-agent.svg?style=social&label=Star) <br> [**Anywhere: A Multi-Agent Framework for User-Guided, Reliable, and Diverse Foreground-Conditioned Image Generation**](https://arxiv.org/abs/2404.18598) <br> | AAAI | 2024-04 | [Github](https://github.com/Sealical/anywhere-multi-agent) |
| [**CURVE: CLIP-Utilized Reinforcement Learning for Visual Image Enhancement via Simple Image Processing**](https://arxiv.org/abs/2505.23102) <br> | ICIP | 2025-05 | [-](-) |
| ![Star](https://img.shields.io/github/stars/Junboooo/RealSR-R1.svg?style=social&label=Star) <br> [**RealSR-R1: Reinforcement Learning for Real-World Image Super-Resolution with Vision-Language Chain-of-Thought**](https://arxiv.org/abs/2506.16796) <br> | arXiv | 2025-06 | [Github](https://github.com/Junboooo/RealSR-R1) |
| ![Star](https://img.shields.io/github/stars/taco-group/4KAgent.svg?style=social&label=Star) <br> [**4KAgent: Agentic Any Image to 4K Super-Resolution**](https://arxiv.org/abs/2507.07105) <br> | NeurIPS | 2025-07 | [Github](https://github.com/taco-group/4KAgent) |
| ![Star](https://img.shields.io/github/stars/zhentao-zou/MURE.svg?style=social&label=Star) <br> [**Beyond Textual CoT: Interleaved Text-Image Chains with Deep Confidence Reasoning for Image Editing**](https://arxiv.org/abs/2510.08157) <br> | arXiv | 2025-10 | [Github](https://github.com/zhentao-zou/MURE) |
| [**Dynamic VLM-Guided Negative Prompting for Diffusion Models**](https://arxiv.org/abs/2510.26052) <br> | NeurIPS | 2025-10 | [-](-) |
| ![Star](https://img.shields.io/github/stars/HerzogFL/VisPainter.svg?style=social&label=Star) <br> [**From Pixels to Paths: A Multi-Agent Framework for Editable Scientific Illustration**](https://arxiv.org/abs/2510.27452) <br> | arXiv | 2025-10 | [Github](https://github.com/HerzogFL/VisPainter) |

---

### <a id="3-extended-applications">3. Extended Applications</a>

#### <a id="31-medical-image-processing">3.1 Medical Image Processing</a>

##### <a id="311-biomedical-vlms">3.1.1 Biomedical VLMs</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/FreedomIntelligence/HuatuoGPT-Vision.svg?style=social&label=Star) <br> [**HuatuoGPT-Vision: Towards Injecting Medical Visual Knowledge into Multimodal LLMs at Scale**](https://arxiv.org/abs/2406.19280) <br> | EMNLP | 2024-06 | [Github](https://github.com/FreedomIntelligence/HuatuoGPT-Vision) |
| ![Star](https://img.shields.io/github/stars/RyannChenOO/MLeVLM.svg?style=social&label=Star) <br> [**MLeVLM: Improve Multi-level Progressive Capabilities based on Multimodal Large Language Model for Medical Visual Question Answering**](https://aclanthology.org/2024.findings-acl.296/) <br> | ACL | 2024-08 | [Github](https://github.com/RyannChenOO/MLeVLM) |
| [**PET Image Denoising via Text-Guided Diffusion: Integrating Anatomical Priors through Text Prompts**](https://arxiv.org/abs/2502.21260) <br> | arXiv | 2025-02 | [-](-) |
| [**AgentPolyp: Accurate Polyp Segmentation via Image Enhancement Agent**](https://arxiv.org/abs/2504.10978) <br> | SPL | 2025-04 | [-](-) |

##### <a id="312-ct-and-mri">3.1.2 CT and MRI</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**Controllable Text-to-Image Synthesis for Multi-Modality MR Images**](https://openaccess.thecvf.com/content/WACV2024/html/Kim_Controllable_Text-to-Image_Synthesis_for_Multi-Modality_MR_Images_WACV_2024_paper.html) <br> | WACV | 2024-01 | [-](-) |
| ![Star](https://img.shields.io/github/stars/hao1635/LEDA.svg?style=social&label=Star) <br> [**Low-dose CT Denoising with Language-engaged Dual-space Alignment**](https://arxiv.org/abs/2403.06128) <br> | arXiv | 2024-03 | [Github](https://github.com/hao1635/LEDA) |
| [**Dual-Domain CLIP-Assisted Residual Optimization Perception Model for Metal Artifact Reduction**](https://arxiv.org/abs/2408.14342) <br> | TRPMS | 2024-08 | [-](-) |
| [**A-IDE: Agent-Integrated Denoising Experts**](https://arxiv.org/abs/2503.16780) <br> | arXiv | 2025-03 | [-](-) |
| ![Star](https://img.shields.io/github/stars/Zi-YuanYang/SCAN-PhysFed.svg?style=social&label=Star) <br> [**Patient-Level Anatomy Meets Scanning-Level Physics: Personalized Federated Low-Dose CT Denoising Empowered by Large Language Model**](https://arxiv.org/abs/2503.00908) <br> | CVPR | 2025-03 | [Github](https://github.com/Zi-YuanYang/SCAN-PhysFed) |
| [**TDMF: Text-Guided Denoising and Interactive Medical Image Fusion**](https://doi.org/10.1109/ICASSP49660.2025.10889309) <br> | ICASSP | 2025-04 | [-](-) |
| ![Star](https://img.shields.io/github/stars/MedAITech/SD4CTSR.svg?style=social&label=Star) <br> [**Taming Stable Diffusion for Computed Tomography Blind Super-Resolution**](https://arxiv.org/abs/2506.11496) <br> | MICCAI WS | 2025-06 | [Github](https://github.com/MedAITech/SD4CTSR) |
| ![Star](https://img.shields.io/github/stars/hao1635/LangMamba.svg?style=social&label=Star) <br> [**LangMamba: A Language-driven Mamba Framework for Low-dose CT Denoising with Vision-language Models**](https://arxiv.org/abs/2507.06140) <br> | TRPMS | 2025-07 | [Github](https://github.com/hao1635/LangMamba) |
| ![Star](https://img.shields.io/github/stars/itu-biai/medsiglip_ldct_iqa.svg?style=social&label=Star) <br> [**Prompt-Conditioned FiLM and Multi-Scale Fusion on MedSigLIP for Low-Dose CT Quality Assessment**](https://arxiv.org/abs/2511.12256) <br> | arXiv | 2025-11 | [Github](https://github.com/itu-biai/medsiglip_ldct_iqa) |

#### <a id="32-remote-sensing-data-processing">3.2 Remote Sensing Data Processing</a>

##### <a id="321-spatial-domain-tasks">3.2.1 Spatial-domain Tasks</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
|[**Large Foundation Model Empowered Discriminative Underwater Image Enhancement**](https://ieeexplore.ieee.org/document/10824846) <br> | IEEE | 2025-01 | [Gitee](https://gitee.com/wanghaoupc/UIE_SAM) |
| [**Semantic-Aware Guidance for Blind Super-Resolution of Remote Sensing Images**](https://dblp.org/rec/journals/lgrs/WuHW25.html) <br> | GRSL | 2025-01 | [-](-) |
| ![Star](https://img.shields.io/github/stars/Mr-Bamboo/SeG-SR.svg?style=social&label=Star) <br> [**SeG-SR: Integrating Semantic Knowledge into Remote Sensing Image Super-Resolution via Vision-Language Model**](https://arxiv.org/abs/2505.23010) <br> | TGRS | 2025-05 | [Github](https://github.com/Mr-Bamboo/SeG-SR) |

##### <a id="322-spectral-domain-tasks">3.2.2 Spectral-domain Tasks</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/ZhehuiWu/MP-HSIR.svg?style=social&label=Star) <br> [**MP-HSIR: A Multi-Prompt Framework for Universal Hyperspectral Image Restoration**](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_MP-HSIR_A_Multi-Prompt_Framework_for_Universal_Hyperspectral_Image_Restoration_ICCV_2025_paper.html) <br> | ICCV | 2025-03 | [Github](https://github.com/ZhehuiWu/MP-HSIR) |

#### <a id="33-other-extended-applications">3.3 Other Extended Applications</a>

##### <a id="331-cad">3.3.1 CAD</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**ChatCAD: An MLLM-Guided Framework for Zero-shot CAD Drawing Restoration**](https://dblp.org/rec/conf/icassp/TangXLWG25.html) <br> | ICASSP | 2025-04 | [-](-) |
| ![Star](https://img.shields.io/github/stars/om-ai-lab/ImageRAG.svg?style=social&label=Star) <br> [**ImageRAG: Enhancing Ultra High Resolution Remote Sensing Imagery Analysis with ImageRAG**](https://arxiv.org/abs/2411.07688) <br> | IEEE | 2024-11 | [Github](https://github.com/om-ai-lab/ImageRAG) |
| ![Star](https://img.shields.io/github/stars/adamhazimeh/SliDer.svg?style=social&label=Star) <br> [**Semantic Document Derendering: SVG Reconstruction via Vision-Language Modeling**](https://arxiv.org/abs/2511.13478) <br> | AAAI | 2025-11 | [Github](https://github.com/adamhazimeh/SliDer) |
| [**ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models**](https://arxiv.org/abs/2512.06328) <br> | AAAI | 2025-12 | [-](-) |

##### <a id="332-video-processing-tasks">3.3.2 Video Processing Tasks</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/jianzongwu/Language-Driven-Video-Inpainting.svg?style=social&label=Star) <br> [**Towards Language-Driven Video Inpainting via Multimodal Large Language Models**](https://arxiv.org/abs/2401.10226) <br> | CVPR | 2024-01 | [Github](https://github.com/jianzongwu/Language-Driven-Video-Inpainting) |
| [**Triplane-Smoothed Video Dehazing with CLIP-Enhanced Generalization**](https://link.springer.com/article/10.1007/s11263-024-02161-0) <br> | IJCV | 2024-08 | [-](-) |
| [**Universal Video Face Restoration Method Based on Vision-Language Model**](https://proceedings.mlr.press/v260/xu25a.html) <br> | ACML | 2024-09 | [-](-) |
| [**Towards General-Purpose Video Reconstruction through Synergy of Grid-Splicing Diffusion and Large Language Models**](https://doi.org/10.1109/TCSVT.2025.3545795) <br> | TCSVT | 2025-03 | [-](-) |
| [**Grounding Degradations in Natural Language for All-In-One Video Restoration**](https://arxiv.org/abs/2507.14851) <br> | arXiv | 2025-07 | [-](-) |
| [**When MLLMs Meet Compression Distortion: A Coding Paradigm Tailored to MLLMs**](https://arxiv.org/abs/2509.24258) <br> | arXiv | 2025-09 | [-](-) |
| [**Edit-Your-Interest: Efficient Video Editing via Feature Most-Similar Propagation**](https://arxiv.org/abs/2510.13084) <br> | arXiv | 2025-10 | [-](-) |
| ![Star](https://img.shields.io/github/stars/tvaranka/ZSVD.svg?style=social&label=Star) <br> [**Zero-Shot Video Deraining with Video Diffusion Models**](https://arxiv.org/abs/2511.18537) <br> | WACV | 2025-11 | [Github](https://github.com/tvaranka/ZSVD) |
| ![Star](https://img.shields.io/github/stars/Liuxinyv/ReViSE.svg?style=social&label=Star) <br> [**ReViSE: Towards Reason-Informed Video Editing in Unified Models with Self-Reflective Learning**](https://arxiv.org/abs/2512.09924) <br> | arXiv | 2025-12 | [Github](https://github.com/Liuxinyv/ReViSE) |

##### <a id="333-3d-processing-tasks">3.3.3 3D Processing Tasks</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/facebookresearch/DepthLM_Official.svg?style=social&label=Star) <br> [**DepthLM: Metric Depth From Vision Language Models**](https://arxiv.org/abs/2509.25413) <br> | arXiv | 2025-09 | [Github](https://github.com/facebookresearch/DepthLM_Official) |
| ![Star](https://img.shields.io/github/stars/XinyuanHu66/SRSplat_Code.svg?style=social&label=Star) <br> [**SRSplat: Feed-Forward Super-Resolution Gaussian Splatting from Sparse Multi-View Images**](https://arxiv.org/abs/2511.12040) <br> | AAAI | 2025-11 | [Github](https://github.com/XinyuanHu66/SRSplat_Code) |
| [**GS-Light: Training-Free Multi-View Extension of IC-Light for Textual Position-Aware Scene Relighting**](https://arxiv.org/abs/2511.13684) <br> | arXiv | 2025-11 | [-](-) |
| ![Star](https://img.shields.io/github/stars/TAU-VAILab/Lang3D-XL.svg?style=social&label=Star) <br> [**Lang3D-XL: Language Embedded 3D Gaussians for Large-scale Scenes**](https://arxiv.org/abs/2512.07807) <br> | SIGGRAPH Asia | 2025-12 | [Github](https://github.com/TAU-VAILab/Lang3D-XL) |

---

## <a id="datasets">Datasets</a>

|  Name  |   Paper  |   Link   |   Notes   |
|:--------|:--------:|:--------:|:--------:|
<!-- To be filled -->

---

## <a id="related-surveys-recommended">Related Surveys Recommended</a>

<!-- To be filled -->

---

## <a id="others">Others</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/TaoWangzj/GridFormer.svg?style=social&label=Star) <br> [**GridFormer: Residual Dense Transformer with Grid Structure for Image Restoration in Adverse Weather Conditions**](https://arxiv.org/abs/2305.17863) <br> | IJCV | 2023-05 | [Github](https://github.com/TaoWangzj/GridFormer) |
| ![Star](https://img.shields.io/github/stars/xinwei666/MMGenerativeIR.svg?style=social&label=Star) <br> [**Generative Multi-Modal Knowledge Retrieval with Large Language Models**](https://arxiv.org/abs/2401.08206) <br> | AAAI | 2024-01 | [Github](https://github.com/xinwei666/MMGenerativeIR) |
| ![Star](https://img.shields.io/github/stars/ByungKwanLee/CoLLaVO.svg?style=social&label=Star) <br> [**CoLLaVO: Crayon Large Language and Vision mOdel**](https://arxiv.org/abs/2402.11248) <br> | arXiv | 2024-02 | [Github](https://github.com/ByungKwanLee/CoLLaVO) |
| ![Star](https://img.shields.io/github/stars/liyongqi67/GRACE.svg?style=social&label=Star) <br> [**Generative Cross-Modal Retrieval: Memorizing Images in Multimodal Language Models for Retrieval and Beyond**](https://aclanthology.org/2024.acl-long.639/) <br> | ACL | 2024-02 | [Github](https://github.com/liyongqi67/GRACE) |
| ![Star](https://img.shields.io/github/stars/microsoft/MMCTAgent.svg?style=social&label=Star) <br> [**MMCTAgent: Multi-modal Critical Thinking Agent Framework for Complex Visual Reasoning**](https://arxiv.org/abs/2405.18358) <br> | NeurIPSW | 2024-05 | [Github](https://github.com/microsoft/MMCTAgent) |
| ![Star](https://img.shields.io/github/stars/lxtGH/OMG-Seg.svg?style=social&label=Star) <br> [**OMG-LLaVA: Bridging Image-level, Object-level, Pixel-level Reasoning and Understanding**](https://arxiv.org/abs/2406.19389) <br> | NeurIPS | 2024-06 | [Github](https://github.com/lxtGH/OMG-Seg) |
| ![Star](https://img.shields.io/github/stars/zhoushen1/MEASNet.svg?style=social&label=Star) <br> [**Multi-Expert Adaptive Selection: Task-Balancing for All-in-One Image Restoration**](https://arxiv.org/abs/2407.19139) <br> | arXiv | 2024-07 | [Github](https://github.com/zhoushen1/MEASNet) |
| ![Star](https://img.shields.io/github/stars/anguyen8/vision-llms-are-blind.svg?style=social&label=Star) <br> [**Vision Language Models Are Blind: Failing to Translate Detailed Visual Features into Words**](https://arxiv.org/abs/2407.06581) <br> | ACCV | 2024-07 | [Github](https://github.com/anguyen8/vision-llms-are-blind) |
| ![Star](https://img.shields.io/github/stars/chxy95/GenLV.svg?style=social&label=Star) <br> [**Learning A Low-Level Vision Generalist via Visual Task Prompt**](https://arxiv.org/abs/2408.08601) <br> | ACM MM | 2024-08 | [Github](https://github.com/chxy95/GenLV) |
| [**Scene Text Detection in Foggy Weather Utilizing Knowledge Distillation of Diffusion Models**](https://doi.org/10.1109/LSP.2025.3540371) <br> | SPL | 2025-02 | [-](-) |
| ![Star](https://img.shields.io/github/stars/wisper12933/GA-Rollback.svg?style=social&label=Star) <br> [**Generator-Assistant Stepwise Rollback Framework for Large Language Model Agent**](https://arxiv.org/abs/2503.02519) <br> | arXiv | 2025-03 | [Github](https://github.com/wisper12933/GA-Rollback) |
| [**Why Compress What You Can Generate? When GPT-4o Generation Ushers in Image Compression Fields**](https://arxiv.org/abs/2504.21814) <br> | ICCVW | 2025-04 | [-](-) |
| ![Star](https://img.shields.io/github/stars/HaoZhang1018/OmniFuse.svg?style=social&label=Star) <br> [**OmniFuse: Composite Degradation-Robust Image Fusion with Language-Driven Semantics**](https://doi.org/10.1109/TPAMI.2025.3568433) <br> | TPAMI | 2025-05 | [Github](https://github.com/HaoZhang1018/OmniFuse) |
| [**Deformable Attentive Visual Enhancement for Referring Segmentation Using Vision-Language Model**](https://arxiv.org/abs/2505.19242) <br> | arXiv | 2025-05 | [-](-) |
| [**Vision-Language Model Priors-Driven State Space Model for Infrared-Visible Image Fusion**](https://doi.org/10.1109/LSP.2025.3578250) <br> | SPL | 2025-06 | [-](-) |
| [**UniLDiff: Unlocking the Power of Diffusion Priors for All-in-One Image Restoration**](https://arxiv.org/abs/2507.23685) <br> | arXiv | 2025-07 | [-](-) |
| [**Follow-Your-Instruction: A Comprehensive MLLM Agent for World Data Synthesis**](https://arxiv.org/abs/2508.05580) <br> | arXiv | 2025-08 | [-](-) |
| ![Star](https://img.shields.io/github/stars/sunwoocho/SRDD.svg?style=social&label=Star) <br> [**Dataset Distillation for Super-Resolution without Class Labels and Pre-trained Models**](https://arxiv.org/abs/2509.14777) <br> | arXiv | 2025-09 | [Github](https://github.com/sunwoocho/SRDD) |
| [**Evaluating Robustness of Vision-Language Models Under Noisy Conditions**](https://arxiv.org/abs/2509.12492) <br> | arXiv | 2025-09 | [-](-) |
| [**VLM-Guided Inpainting for Anomaly Detection**](https://doi.org/10.33851/JMIS.2025.12.3.87) <br> | Journal of Multimedia Information System | 2025-09 | [-](-) |
| [**Clear Roads, Clear Vision: Advancements in Multi-Weather Restoration for Smart Transportation**](https://arxiv.org/abs/2510.09228) <br> | arXiv | 2025-10 | [Github](https://github.com/ChaudharyUPES/) |
| ![Star](https://img.shields.io/github/stars/RitAreaSciencePark/physics-informed-stm-restoration.svg?style=social&label=Star) <br> [**Generative Image Restoration and Super-Resolution using Physics-Informed Synthetic Data for Scanning Tunneling Microscopy**](https://arxiv.org/abs/2510.25921) <br> | arXiv | 2025-10 | [Github](https://github.com/RitAreaSciencePark/physics-informed-stm-restoration) |
| [**DiffSeg30k: A Multi-Turn Diffusion Editing Benchmark for Localized AIGC Detection**](https://arxiv.org/abs/2511.19111) <br> | arXiv | 2025-11 | [HuggingFace](https://huggingface.co/datasets/Chaos2629/Diffseg30k) |

---

## <a id="reference">Reference</a>
[Awesome-Multimodal-Large-Language-Models-by-BradyFU](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models-by-BradyFU)
