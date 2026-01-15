# <p align=center>Awesome Multimodal Large Language Models In Low-level Vision[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/ChunmingHe/awesome-multimodal-large-language-models-in-low-level-vision)</p>

<p align=center>🔥A curated list of awesome <b>Multimodal Large Language Models(MLLMs)</b> & <b>Vision-Language Models(MLLMs)</b> in low-level vision.🔥</p>

<p align=center>Please feel free to offer your suggestions in the Issues and pull requests to add links.</p>

<p align=center><b>[ Last updated at 2026/01/14 ]</b></p>

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

---

### <a id="1-direct-vlm-adaptation-for-low-level-vision">1. Direct VLM Adaptation for Low-Level Vision</a>

#### <a id="11-visual-encoder-adaptation-handling-details">1.1 Visual Encoder Adaptation: Handling Details</a>

##### <a id="111-resolution-scaling">1.1.1 Resolution Scaling</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/icandle/GenDR.svg?style=social&label=Star) <br> [**GenDR: Lightning Generative Detail Restorator**](https://arxiv.org/abs/2503.06790) <br> | arXiv | 2025-03 | [Github](https://github.com/icandle/GenDR) |

##### <a id="112-feature-fusion">1.1.2 Feature Fusion</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|

#### <a id="12-language-branch-adaptation-bridging-modalities">1.2 Language Branch Adaptation: Bridging Modalities</a>

##### <a id="121-prompt-learning-strategies">1.2.1 Prompt Learning Strategies</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|

##### <a id="122-instruction-tuning-for-restoration">1.2.2 Instruction Tuning for Restoration</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**Lumina-OmniLV: A Unified Multimodal Framework for General Low-Level Vision**](https://arxiv.org/abs/2504.04903) <br> | arXiv | 2025-04 | [-](-) |

#### <a id="13-output-head-adaptation-from-tokens-to-pixels">1.3 Output Head Adaptation: From Tokens to Pixels</a>

##### <a id="131-tokenizerdecoder-framework">1.3.1 Tokenizer–Decoder Framework</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/nonwhy/PURE.svg?style=social&label=Star) <br> [**Perceive, Understand and Restore: Real-World Image Super-Resolution with Autoregressive Multimodal Generative Models**](https://openaccess.thecvf.com/content/ICCV2025/html/Wei_Perceive_Understand_and_Restore_Real-World_Image_Super-Resolution_with_Autoregressive_Multimodal_ICCV_2025_paper.html) <br> | ICCV | 2025-03 | [Github](https://github.com/nonwhy/PURE) |
| [**SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook for Multimodal Understanding and Generation**](http://arxiv.org/abs/2503.06764) <br> | arXiv | 2025-03 | [-](-) |

#### <a id="14-parameter-efficient-fine-tuning-in-restoration">1.4 Parameter-Efficient Fine-Tuning in Restoration</a>

##### <a id="141-lora--adapter-integration">1.4.1 LoRA & Adapter Integration</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**Acquire and then Adapt: Squeezing out Text-to-Image Model for Image Restoration**](http://arxiv.org/abs/2504.15159) <br> | CVPR | 2025-04 | [-](-) |

##### <a id="142-freezing-strategies">1.4.2 Freezing Strategies</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|

---

### <a id="2-vlm-as-auxiliary-for-low-level-vision">2. VLM as Auxiliary for Low-Level Vision</a>

#### <a id="21-vlm-as-semantic-provider-text-guided-restoration">2.1 VLM as Semantic Provider: Text-Guided Restoration</a>

##### <a id="211-text-conditioned-injection">2.1.1 Text-Conditioned Injection</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/zhaolb4080/MTG-Fusion.svg?style=social&label=Star) <br> [**Multi-Text Guidance Is Important: Multi-Modality Image Fusion via Large Generative Vision-Language Model**](https://doi.org/10.1007/s11263-025-02409-3) <br> | IJCV | 2025-02 | [Github](https://github.com/zhaolb4080/MTG-Fusion) |
| ![Star](https://img.shields.io/github/stars/striveAgain/MegaSR.svg?style=social&label=Star) <br> [**MegaSR: Mining Customized Semantics and Expressive Guidance for Image Super-Resolution**](https://arxiv.org/abs/2503.08096) <br> | arXiv | 2025-03 | [Github](https://github.com/striveAgain/MegaSR) |
| [**The Power of Context: How Multimodality Improves Image Super-Resolution**](https://arxiv.org/abs/2503.14503) <br> | CVPR | 2025-03 | [-](-) |
| [**Coarse-to-fine text injecting for realistic image super-resolution**](https://doi.org/10.1016/j.neucom.2025.129591) <br> | Neurocom | 2025-02 | [-](-) |
| ![Star](https://img.shields.io/github/stars/Wenyuzhy/DeepSPG.svg?style=social&label=Star) <br> [**DeepSPG: Exploring Deep Semantic Prior Guidance for Low-light Image Enhancement with Multimodal Learning**](https://arxiv.org/abs/2504.19127) <br> | arXiv | 2025-04 | [Github](https://github.com/Wenyuzhy/DeepSPG) |
| ![Star](https://img.shields.io/github/stars/ywxjm/Diff-Dehazer.svg?style=social&label=Star) <br> [**Exploiting Diffusion Prior for Real-World Image Dehazing with Unpaired Training**](https://arxiv.org/abs/2503.15017) <br> | AAAI | 2025-03 | [Github](https://github.com/ywxjm/Diff-Dehazer) |

##### <a id="212-language-driven-manipulation">2.1.2 Language-Driven Manipulation</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/SherryXTChen/Instruct-CLIP.svg?style=social&label=Star) <br> [**Instruct-CLIP: Improving Instruction-Guided Image Editing with Automated Data Refinement Using Contrastive Learning**](https://arxiv.org/abs/2503.18406) <br> | CVPR | 2025-03 | [Github](https://github.com/SherryXTChen/Instruct-CLIP) |

##### <a id="213-subject-aware-restoration">2.1.3 Subject-Aware Restoration</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/shuaizhengliu/InstructRestore.svg?style=social&label=Star) <br> [**InstructRestore: Region-Customized Image Restoration with Human Instructions**](https://arxiv.org/abs/2503.24357) <br> | arXiv | 2025-03 | [Github](https://github.com/shuaizhengliu/InstructRestore) |
| [**PromptLNet: Region-Adaptive Aesthetic Enhancement via Prompt Guidance in Low-Light Enhancement Net**](https://arxiv.org/abs/2503.08276) <br> | arXiv | 2025-03 | [-](-) |
| [**TSCnet: A Text-driven Semantic-level Controllable Framework for Customized Low-Light Image Enhancement**](http://arxiv.org/abs/2503.08168) <br> | Neurocomputing | 2025-03 | [Project](https://miaorain.github.io/lowlight09.github.io/) |

#### <a id="22-vlm-as-degradation-interpreter-visual-prompting--context">2.2 VLM as Degradation Interpreter: Visual-Prompting & Context</a>

##### <a id="221-degradation-classification">2.2.1 Degradation Classification</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**Multi-modal degradation feature learning for unified image restoration based on contrastive learning**](https://doi.org/10.1016/j.neucom.2024.128955) <br> | Neurocomputing | 2024-11 | [-](-) |
| ![Star](https://img.shields.io/github/stars/xianggkl/VLU-Net.svg?style=social&label=Star) <br> [**Vision-Language Gradient Descent-driven All-in-One Deep Unfolding Networks**](https://arxiv.org/abs/2503.16930) <br> | CVPR | 2025-03 | [Github](https://github.com/xianggkl/VLU-Net) |
| [**DA2Diff: Exploring Degradation-aware Adaptive Diffusion Priors for All-in-One Weather Restoration**](https://arxiv.org/abs/2504.05135) <br> | arXiv | 2025-04 | [-](-) |
| [**VL-UR: Vision-Language-guided Universal Restoration of Images Degraded by Adverse Weather Conditions**](https://arxiv.org/abs/2504.08219) <br> | ICME | 2025-04 | [-](-) |

##### <a id="222-description-based-restoration">2.2.2 Description-based Restoration</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/sudraj2002/AWRaCLe.svg?style=social&label=Star) <br> [**AWRaCLe: All-Weather Image Restoration using Visual In-Context Learning**](https://arxiv.org/abs/2409.00263) <br> | AAAI | 2024-08 | [Github](https://github.com/sudraj2002/AWRaCLe) |
| [**Leveraging vision-language prompts for real-world image restoration and enhancement**](https://www.sciencedirect.com/science/article/abs/pii/S1077314224003035) <br> | CVIU | 2024-11 | [-](-) |
| ![Star](https://img.shields.io/github/stars/igor-morawski/tmm-sem.svg?style=social&label=Star) <br> [**Leveraging Content and Context Cues for Low-Light Image Enhancement**](https://arxiv.org/abs/2412.07693) <br> | TMM | 2024-12 | [Github](https://github.com/igor-morawski/tmm-sem) |
| [**RamIR: Reasoning and action prompting with Mamba for all-in-one image restoration**](https://link.springer.com/article/10.1007/s10489-024-06226-y) <br> | - | 2025-01 | [-](-) |
| ![Star](https://img.shields.io/github/stars/Linfeng-Tang/ControlFusion.svg?style=social&label=Star) <br> [**ControlFusion: A Controllable Image Fusion Network with Language-Vision Degradation Prompts**](https://arxiv.org/abs/2503.23356) <br> | arXiv | 2025-03 | [Github](https://github.com/Linfeng-Tang/ControlFusion) |
| ![Star](https://img.shields.io/github/stars/RongxinL/CyclicPrompt.svg?style=social&label=Star) <br> [**Prompt to Restore, Restore to Prompt: Cyclic Prompting for Universal Adverse Weather Removal**](https://arxiv.org/abs/2503.09013) <br> | TIP | 2025-03 | [Github](https://github.com/RongxinL/CyclicPrompt) |
| [**UniCoRN: Latent Diffusion-based Unified Controllable Image Restoration Network across Multiple Degradations**](https://arxiv.org/abs/2503.15868) <br> | WACV (Acc.) | 2025-03 | [Project](https://codejaeger.github.io/unicorn-gh/) |
| ![Star](https://img.shields.io/github/stars/HXDreamChaser/CLIP-RestoreX.svg?style=social&label=Star) <br> [**CLIP-RestoreX: Restore Image Structure and Perception in Exposure Correction**](https://ojs.aaai.org/index.php/AAAI/article/view/32392) <br> | AAAI | 2025-04 | [Github](https://github.com/HXDreamChaser/CLIP-RestoreX) |
| ![Star](https://img.shields.io/github/stars/kongdehong/DPIR.svg?style=social&label=Star) <br> [**Dual Prompting Image Restoration with Diffusion Transformers**](https://openaccess.thecvf.com/content/CVPR2025/html/Kong_Dual_Prompting_Image_Restoration_with_Diffusion_Transformers_CVPR_2025_paper.html) <br> | CVPR | 2025-04 | [Github](https://github.com/kongdehong/DPIR) |

#### <a id="23-vlm-as-quality-evaluator-perception-and-assessment">2.3 VLM as Quality Evaluator: Perception and Assessment</a>

##### <a id="231-no-reference-quality-assessment">2.3.1 No-Reference Quality Assessment</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/LowLevelAI/GPP-LLIE.svg?style=social&label=Star) <br> [**Low-Light Image Enhancement via Generative Perceptual Priors**](https://arxiv.org/abs/2412.20916) <br> | AAAI | 2024-12 | [Github](https://github.com/LowLevelAI/GPP-LLIE) |
| ![Star](https://img.shields.io/github/stars/zhiyuanyou/DeQA-Score.svg?style=social&label=Star) <br> [**Teaching Large Language Models to Regress Accurate Image Quality Scores using Score Distribution**](https://arxiv.org/abs/2501.11561) <br> | CVPR | 2025-01 | [Github](https://github.com/zhiyuanyou/DeQA-Score) |
| ![Star](https://img.shields.io/github/stars/JunFu1995/CLIP-DQA.svg?style=social&label=Star) <br> [**CLIP-DQA: Blindly Evaluating Dehazed Images from Global and Local Perspectives Using CLIP**](https://arxiv.org/abs/2502.01707) <br> | ISCAS | 2025-02 | [Github](https://github.com/JunFu1995/CLIP-DQA) |
| [**DVLTA-VQA: Decoupled Vision-Language Modeling with Text-Guided Adaptation for Blind Video Quality Assessment**](https://arxiv.org/abs/2504.11733) <br> | arXiv | 2025-04 | [-](-) |
| ![Star](https://img.shields.io/github/stars/yeppp27/Q-Adapt.svg?style=social&label=Star) <br> [**Q-Adapt: Adapting LMM for Visual Quality Assessment with Progressive Instruction Tuning**](https://arxiv.org/abs/2504.01655) <br> | arXiv | 2025-04 | [Github](https://github.com/yeppp27/Q-Adapt) |

##### <a id="232-semantic-consistency-loss">2.3.2 Semantic Consistency Loss</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|

##### <a id="233-feedback-loops">2.3.3 Feedback Loops</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**DSPO: Direct Semantic Preference Optimization for Real-World Image Super-Resolution**](https://arxiv.org/abs/2504.15176) <br> | arXiv | 2025-04 | [-](-) |
| ![Star](https://img.shields.io/github/stars/alexlai2860/SnowMaster.svg?style=social&label=Star) <br> [**SnowMaster: Comprehensive Real-world Image Desnowing via MLLM with Multi-Model Feedback Optimization**](https://openaccess.thecvf.com/content/CVPR2025/html/Lai_SnowMaster_Comprehensive_Real-world_Image_Desnowing_via_MLLM_with_Multi-Model_Feedback_CVPR_2025_paper.html) <br> | CVPR | 2025-06 | [Github](https://github.com/alexlai2860/SnowMaster) |

#### <a id="24-vlm-as-intelligent-controller-agent-based-frameworks">2.4 VLM as Intelligent Controller: Agent-Based Frameworks</a>

##### <a id="241-motivation-as-intelligent-controller">2.4.1 Motivation as Intelligent Controller</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|

##### <a id="242-tool-usage--orchestration">2.4.2 Tool Usage & Orchestration</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/cilabuniba/i-dream-my-painting.svg?style=social&label=Star) <br> [**I Dream My Painting: Connecting MLLMs and Diffusion Models via Prompt Generation for Text-Guided Multi-Mask Inpainting**](https://arxiv.org/abs/2411.19050) <br> | WACV | 2024-11 | [Github](https://github.com/cilabuniba/i-dream-my-painting) |
| [**Hybrid Agents for Image Restoration**](https://arxiv.org/abs/2503.10120) <br> | arXiv | 2025-03 | [-](-) |
| [**Multi-Agent Image Restoration**](https://arxiv.org/abs/2503.09403) | arXiv | 2025-03 | [-](-) |
| ![Star](https://img.shields.io/github/stars/LYL1015/JarvisIR.svg?style=social&label=Star) <br> [**JarvisIR: Elevating Autonomous Driving Perception with Intelligent Image Restoration**](https://arxiv.org/abs/2504.04158) <br> | CVPR | 2025-04 | [Github](https://github.com/LYL1015/JarvisIR) |
| [**Q-Agent: Quality-Driven Chain-of-Thought Image Restoration Agent through Robust Multimodal Large Language Model**](https://arxiv.org/abs/2504.07148) <br> | arXiv | 2025-04 | [-](-) |

##### <a id="243-iterative-refinement">2.4.3 Iterative Refinement</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|

---

### <a id="3-extended-applications">3. Extended Applications</a>

#### <a id="31-medical-image-processing">3.1 Medical Image Processing</a>

##### <a id="311-biomedical-vlms">3.1.1 Biomedical VLMs</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**PET Image Denoising via Text-Guided Diffusion: Integrating Anatomical Priors through Text Prompts**](https://arxiv.org/abs/2502.21260) <br> | arXiv | 2025-02 | [-](-) |
| [**AgentPolyp: Accurate Polyp Segmentation via Image Enhancement Agent**](https://arxiv.org/abs/2504.10978) <br> | SPL | 2025-04 | [-](-) |

##### <a id="312-ct-and-mri">3.1.2 CT and MRI</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**A-IDE: Agent-Integrated Denoising Experts**](https://arxiv.org/abs/2503.16780) <br> | arXiv | 2025-03 | [-](-) |
| ![Star](https://img.shields.io/github/stars/Zi-YuanYang/SCAN-PhysFed.svg?style=social&label=Star) <br> [**Patient-Level Anatomy Meets Scanning-Level Physics: Personalized Federated Low-Dose CT Denoising Empowered by Large Language Model**](https://arxiv.org/abs/2503.00908) <br> | CVPR | 2025-03 | [Github](https://github.com/Zi-YuanYang/SCAN-PhysFed) |
| [**TDMF: Text-Guided Denoising and Interactive Medical Image Fusion**](https://doi.org/10.1109/ICASSP49660.2025.10889309) <br> | ICASSP | 2025-04 | [-](-) |

#### <a id="32-remote-sensing-data-processing">3.2 Remote Sensing Data Processing</a>

##### <a id="321-spatial-domain-tasks">3.2.1 Spatial-domain Tasks</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
|[**Large Foundation Model Empowered Discriminative Underwater Image Enhancement**](https://ieeexplore.ieee.org/document/10824846) <br> | IEEE | 2025-01 | [Gitee](https://gitee.com/wanghaoupc/UIE_SAM) |
| [**Semantic-Aware Guidance for Blind Super-Resolution of Remote Sensing Images**](https://dblp.org/rec/journals/lgrs/WuHW25.html) <br> | GRSL | 2025-01 | [-](-) |

##### <a id="322-spectral-domain-tasks">3.2.2 Spectral-domain Tasks</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/ZhehuiWu/MP-HSIR.svg?style=social&label=Star) <br> [**MP-HSIR: A Multi-Prompt Framework for Universal Hyperspectral Image Restoration**](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_MP-HSIR_A_Multi-Prompt_Framework_for_Universal_Hyperspectral_Image_Restoration_ICCV_2025_paper.html) <br> | ICCV | 2025-03 | [Github](https://github.com/ZhehuiWu/MP-HSIR) |

#### <a id="33-other-extended-applications">3.3 Other Extended Applications</a>

##### <a id="331-cad">3.3.1 CAD</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**ChatCAD: An MLLM-Guided Framework for Zero-shot CAD Drawing Restoration**](https://dblp.org/rec/conf/icassp/TangXLWG25.html) <br> | ICASSP | 2025-04 | [-](-) |

##### <a id="332-video-processing-tasks">3.3.2 Video Processing Tasks</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**Towards General-Purpose Video Reconstruction through Synergy of Grid-Splicing Diffusion and Large Language Models**](https://doi.org/10.1109/TCSVT.2025.3545795) <br> | TCSVT | 2025-03 | [-](-) |

##### <a id="333-3d-processing-tasks">3.3.3 3D Processing Tasks</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|

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
| [**Scene Text Detection in Foggy Weather Utilizing Knowledge Distillation of Diffusion Models**](https://doi.org/10.1109/LSP.2025.3540371) <br> | SPL | 2025-02 | [-](-) |
| ![Star](https://img.shields.io/github/stars/wisper12933/GA-Rollback.svg?style=social&label=Star) <br> [**Generator-Assistant Stepwise Rollback Framework for Large Language Model Agent**](https://arxiv.org/abs/2503.02519) <br> | arXiv | 2025-03 | [Github](https://github.com/wisper12933/GA-Rollback) |
| [**Why Compress What You Can Generate? When GPT-4o Generation Ushers in Image Compression Fields**](https://arxiv.org/abs/2504.21814) <br> | ICCVW | 2025-04 | [-](-) |

---

## <a id="reference">Reference</a>
[Awesome-Multimodal-Large-Language-Models-by-BradyFU](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models-by-BradyFU)
