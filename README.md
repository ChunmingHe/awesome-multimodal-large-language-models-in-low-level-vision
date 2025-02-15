
# <p align=center>Awesome Multimodal Large Language Models In Low-level Vision[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/ChunmingHe/awesome-multimodal-large-language-models-in-low-level-vision)</p>

<p align=center>🔥A curated list of awesome <b>Multimodal Large Language Models(MLLMs)</b> & <b>Vision-Language Models(MLLMs)</b> in low-level vision.🔥</p>

<p align=center>Please feel free to offer your suggestions in the Issues and pull requests to add links.</p>

<p align=center><b>[ Last updated at 2024/11/10 ]</b></p>

## Contents

- [Awesome Multimodal Large Language Models In Low-level Vision](#awesome-multimodal-large-language-models-in-low-level-vision)
  - [Contents](#contents)
  - [Latest Works Recommended](#latest-works-recommended)
  - [Awesome Papers](#awesome-papers)
    - [Specific Task](#specific-task)
      - [Denoising](#denoising)
      - [Inpainting](#inpainting)
      - [SR](#sr)
    - [Multiple Tasks](#multiple-tasks)
    - [Other Task](#other-task)
    - [Model Training](#model-training)
      - [Pre Training](#pre-training)
      - [Fine Tuning](#fine-tuning)
  - [Related Surveys Recommended](#related-surveys-recommended)
  - [Benchmarks for Evaluation](#benchmarks-for-evaluation)
    - [Metrics](#metrics)
  - [Reference](#reference)

## <a id="latest-works-recommended">Latest Works Recommended</a>

**Diffusion Models in Low-Level Vision: A Survey**<br />*Chunming He, Yuqi Shen, Chengyu Fang, Fengyang Xiao, Longxiang Tang, Yulun Zhang, Wangmeng Zuo, Zhenhua Guo, Xiu Li*<br />TPAMI, minor revision. [[Paper](https://arxiv.org/abs/2406.11138)] 

**Reti-Diff: Illumination Degradation Image Restoration with Retinex-based Latent Diffusion Model**<br />*Chunming He, Chengyu Fang, Yulun Zhang, Kai Li, Longxiang Tang, Chenyu You, Fengyang Xiao, Zhenhua Guo, Xiu Li*<br />
ICLR 2025, Spotlight. [[Paper](https://arxiv.org/abs/2311.11638)] [[Github](https://github.com/ChunmingHe/Reti-Diff)]<br />
Jan. 2025<br />

## <a id="awesome-papers">Awesome Papers</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/miv-xjtu/flame.svg?style=social&label=Star) <br> [**FLAME: Frozen Large Language Models Enable Data-Efficient Language-Image Pre-training**](http://arxiv.org/abs/2411.11927) <br> | arXiv | 2024-11 | [Github](https://github.com/miv-xjtu/flame) |
| ![Star](https://img.shields.io/github/stars/lyh-18/DegAE_DegradationAutoencoder.svg?style=social&label=Star) <br> [**DegAE: A New Pretraining Paradigm for Low-level Vision**](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_DegAE_A_New_Pretraining_Paradigm_for_Low-Level_Vision_CVPR_2023_paper.html) <br> | CVPR | 2023 | [Github](https://github.com/lyh-18/DegAE_DegradationAutoencoder) |
| [**On Domain-Specific Post-Training for Multimodal Large Language Models**](http://arxiv.org/abs/2411.19930) <br> | arXiv | 2024-11 | [-](-) |
| ![Star](https://img.shields.io/github/stars/x-plug/mplug-owl.svg?style=social&label=Star) <br> [**mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration**](https://openaccess.thecvf.com/content/CVPR2024/html/Ye_mPLUG-Owl2_Revolutionizing_Multi-modal_Large_Language_Model_with_Modality_Collaboration_CVPR_2024_paper.html) <br> | CVPR | 2024 | [Github](https://github.com/x-plug/mplug-owl) |
| ![Star](https://img.shields.io/github/stars/nvlabs/prismer.svg?style=social&label=Star) <br> [**Prismer: A Vision-Language Model with Multi-Task Experts**](http://arxiv.org/abs/2303.02506) <br> | arXiv | 2024-01 | [Github](https://github.com/nvlabs/prismer) |
| ![Star](https://img.shields.io/github/stars/ByungKwanLee/CoLLaVO.svg?style=social&label=Star) <br> [**CoLLaVO: Crayon Large Language and Vision mOdel**](http://arxiv.org/abs/2402.11248) <br> | arXiv | 2024-06 | [Github](https://github.com/ByungKwanLee/CoLLaVO) |
| ![Star](https://img.shields.io/github/stars/lxtgh/omg-seg.svg?style=social&label=Star) <br> [**OMG-LLaVA: Bridging Image-level, Object-level, Pixel-level Reasoning and Understanding**](http://arxiv.org/abs/2406.19389) <br> | arXiv | 2024-10 | [Github](https://github.com/lxtgh/omg-seg) |
| [**Vision language models are blind**](http://arxiv.org/abs/2407.06581) <br>| - | 2024-07 | [-](-) |
| [**ForgeryGPT: Multimodal Large Language Model For Explainable Image Forgery Detection and Localization**](http://arxiv.org/abs/2410.10238) <br> | arXiv | 2024-10 | [-](-) |
| [**A Comprehensive Review of Multimodal Large Language Models: Performance and Challenges Across Different Tasks**](https://arxiv.org/abs/2408.01319) <br> | arXiv | 2024-08 | [-](-) |
| [**Decoder-Only Transformers: The Brains Behind Generative AI, Large Language Models and Large Multimodal Models**](https://www.techrxiv.org/doi/full/10.36227/techrxiv.173198819.91727188) <br> | techrxiv | 2024-11 | [-](-) |
| ![Star](https://img.shields.io/github/stars/ugorsahin/enhancing-multimodal-compositional-reasoning-of-vlm.svg?style=social&label=Star) <br> [**Enhancing Multimodal Compositional Reasoning of Visual Language Models with Generative Negative Mining**](https://openaccess.thecvf.com/content/WACV2024/html/Sahin_Enhancing_Multimodal_Compositional_Reasoning_of_Visual_Language_Models_With_Generative_WACV_2024_paper.html) <br> | WACV | 2024-01 | [Github](https://ugorsahin.github.io/enhancing-multimodal-compositional-reasoning-of-vlm.html) |
| ![Star](https://img.shields.io/github/stars/kohjingyu/gill.svg?style=social&label=Star) <br> [**Generating Images with Multimodal Language Models**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/43a69d143273bd8215578bde887bb552-Abstract-Conference.html) <br> | NeurIPS | 2023-12 | [Github](https://github.com/kohjingyu/gill) |
| [**Generative Cross-Modal Retrieval: Memorizing Images in Multimodal Language Models for Retrieval and Beyond**](https://arxiv.org/abs/2402.10805) <br> | arXiv | 2024-02 | [-](-) |
| ![Star](https://img.shields.io/github/stars/xinwei666/mmgenerativeir.svg?style=social&label=Star) <br> [**Generative Multi-Modal Knowledge Retrieval with Large Language Models**](https://ojs.aaai.org/index.php/AAAI/article/view/29837) <br> | AAAI | 2024-03 | [Github](https://github.com/xinwei666/mmgenerativeir) |
| ![Star](https://img.shields.io/github/stars/baaivision/emu.svg?style=social&label=Star) <br> [**Generative Multimodal Models are In-Context Learners**](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Generative_Multimodal_Models_are_In-Context_Learners_CVPR_2024_paper.html) <br> | CVPR | 2024-06 | [Github](https://github.com/baaivision/emu) |
| ![Star](https://img.shields.io/github/stars/kohjingyu/fromage.svg?style=social&label=Star) <br> [**Grounding Language Models to Images for Multimodal Inputs and Outputs**](https://proceedings.mlr.press/v202/koh23a.html) <br> | PMLR | 2023 | [Github](https://github.com/kohjingyu/fromage) |
| [**Incorporating Visual Experts to Resolve the Information Loss in Multimodal Large Language Models**](https://arxiv.org/abs/2401.03105) <br> | arXiv | 2024-01 | [-](-) |
| [**Q-Instruct: Improving Low-level Visual Abilities for Multi-modality Foundation Models**](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Q-Instruct_Improving_Low-level_Visual_Abilities_for_Multi-modality_Foundation_Models_CVPR_2024_paper.html) <br> | CVPR | 2024-06 | [Github](https://github.com/Q-Future/Q-Instruct) |
| ![Star](https://img.shields.io/github/stars/RLHF-V/RLHF-V.svg?style=social&label=Star) <br> [**RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback**](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_RLHF-V_Towards_Trustworthy_MLLMs_via_Behavior_Alignment_from_Fine-grained_Correctional_CVPR_2024_paper.html) <br> | CVPR | 2024-06 | [Github](https://github.com/RLHF-V/RLHF-V) |
| ![Star](https://img.shields.io/github/stars/karpathy/llama2.c.svg?style=social&label=Star) <br> [**Training Compute-Optimal Large Language Models**](https://arxiv.org/abs/2203.15556) <br> | arXiv | 2022-03 | [Github](https://github.com/karpathy/llama2.c) |

### <a id="specific-task">Specific Task</a>

#### <a id="denoising">Denoising</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/zyhrainbow/SSP-IR.svg?style=social&label=Star) <br> [**MRIR: Integrating Multimodal Insights for Diffusion-based Realistic Image Restoration**](http://arxiv.org/abs/2407.03635) <br> | arXiv | 2024-07 | [GitHub](https://github.com/zyhrainbow/SSP-IR) |
| ![Star](https://img.shields.io/github/stars/FreedomIntelligence/HuatuoGPT-Vision.svg?style=social&label=Star) <br> [**HuatuoGPT-Vision, Towards Injecting Medical Visual Knowledge into Multimodal LLMs at Scale**](http://arxiv.org/abs/2406.19280) <br> | arXiv | 2024-09 | [GitHub](https://github.com/FreedomIntelligence/HuatuoGPT-Vision) |

#### <a id="inpainting">Inpainting</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/dvlab-research/LLMGA.svg?style=social&label=Star) <br> [**LLMGA: Multimodal Large Language Model based Generation Assistant**](http://arxiv.org/abs/2311.16500) <br> | arXiv | 2024-07 | [GitHub](https://github.com/dvlab-research/LLMGA) |
| [**IMProv: Inpainting-based Multimodal Prompting for Computer Vision Tasks**](http://arxiv.org/abs/2312.01771) <br> | arXiv | 2023-12 | [Other](https://jerryxu.net/IMProv/) |
| ![Star](https://img.shields.io/github/stars/jianzongwu/LanguageDriven-Video-Inpainting.svg?style=social&label=Star) <br> [**Towards Language-Driven Video Inpainting via Multimodal Large Language Models**](http://arxiv.org/abs/2401.10226) <br> | arXiv | 2024-10 | [GitHub](https://github.com/jianzongwu/LanguageDriven-Video-Inpainting) |

#### <a id="sr">SR</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/qyp2000/XPSR.svg?style=social&label=Star) <br> [**XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution**](http://arxiv.org/abs/2403.05049) <br> | arXiv | 2024-07 | [GitHub](https://github.com/qyp2000/XPSR) |
| ![Star](https://img.shields.io/github/stars/puppy210/DaLPSR.svg?style=social&label=Star) <br> [**DaLPSR: Leverage Degradation-Aligned Language Prompt for Real-World Image Super-Resolution**](http://arxiv.org/abs/2406.16477) <br> | arXiv | 2024-10 | [GitHub](https://github.com/puppy210/DaLPSR) |

### <a id="multiple-tasks">Multiple Tasks</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/chxy95/GenLV.svg?style=social&label=Star) <br> [**Learning A Low-Level Vision Generalist via Visual Task Prompt**](http://arxiv.org/abs/2408.08601) <br> | arXiv | 2024-08 | [Github](https://github.com/chxy95/GenLV) |
| ![Star](https://img.shields.io/github/stars/bytetriper/LM4LV.svg?style=social&label=Star) <br> [**LM4LV: A Frozen Large Language Model for Low-level Vision Tasks**](http://arxiv.org/abs/2405.15734) <br> | arXiv | 2024-06 | [Github](https://github.com/bytetriper/LM4LV) |
| [**RestoreAgent: Autonomous Image Restoration Agent via Multimodal Large Language Models**](http://arxiv.org/abs/2407.18035) <br> | arXiv | 2024-07 | [Other](https://haoyuchen.com/RestoreAgent) |
| [**LLMRA: Multi-modal Large Language Model based Restoration Assistant**](http://arxiv.org/abs/2401.11401) <br> | arXiv | 2024-01 | [-](-) |
| ![Star](https://img.shields.io/github/stars/zh460045050/V2L-Tokenizer.svg?style=social&label=Star) <br> [**Beyond Text: Frozen Large Language Models in Visual Signal Comprehension**](http://arxiv.org/abs/2403.07874) <br> | arXiv | 2024-03 | [Github](https://github.com/zh460045050/V2L-Tokenizer) |
| ![Star](https://img.shields.io/github/stars/Algolzw/daclip-uir.svg?style=social&label=Star) <br> [**Controlling Vision-Language Models for Multi-Task Image Restoration**](http://arxiv.org/abs/2310.01018) <br> | arXiv | 2024-02 | [Github](https://github.com/Algolzw/daclip-uir) |
| [**Multimodal Prompt Perceiver: Empower Adaptiveness, Generalizability and Fidelity for All-in-One Image Restoration**](http://arxiv.org/abs/2312.02918) <br> | arXiv | 2024-03 | [Github](https://shallowdream204.github.io/mperceiver/) |
| [**Clarity ChatGPT: An Interactive and Adaptive Processing System for Image Restoration and Enhancement**](http://arxiv.org/abs/2311.11695) <br> | arXiv | 2023-11 | [-](-) |
| [**Diff-Restorer: Unleashing Visual Prompts for Diffusion-based Universal Image Restoration**](http://arxiv.org/abs/2407.03636) <br> | arXiv | 2024-07 | [-](-) |
| [**AllRestorer: All-in-One Transformer for Image Restoration under Composite Degradations**](http://arxiv.org/abs/2411.10708) <br> | arXiv | 2024-11 | [-](-) |
| [**GridFormer: Residual Dense Transformer with Grid Structure for Image Restoration in Adverse Weather Conditions**](https://doi.org/10.1007/s11263-024-02056-0) <br> | - | 2024-10 | [-](-) |
| ![Star](https://img.shields.io/github/stars/zhoushen1/MEASNet.svg?style=social&label=Star) <br> [**Multi-Expert Adaptive Selection: Task-Balancing for All-in-One Image Restoration**](http://arxiv.org/abs/2407.19139) <br> | arXiv | 2024-07 | [Github](https://github.com/zhoushen1/MEASNet) |
| [**Leveraging vision-language prompts for real-world image restoration and enhancement**](https://www.sciencedirect.com/science/article/pii/S1077314224003035) <br> | - | 2025-01 | [-](-) |
| ![Star](https://img.shields.io/github/stars/shallowdream204/LoRA-IR.svg?style=social&label=Star) <br> [**LoRA-IR: Taming Low-Rank Experts for Efficient All-in-One Image Restoration**](http://arxiv.org/abs/2410.15385) <br> | arXiv | 2024-11 | [Github](https://github.com/shallowdream204/LoRA-IR) |
| ![Star](https://img.shields.io/github/stars/Kaiwen-Zhu/AgenticIR.svg?style=social&label=Star) <br> [**An Intelligent Agentic System for Complex Image Restoration Problems**](http://arxiv.org/abs/2410.17809) <br> | arXiv | 2024-10 | [Github](https://github.com/Kaiwen-Zhu/AgenticIR) |
| ![Star](https://img.shields.io/github/stars/intmegroup/uniprocessor.svg?style=social&label=Star) <br> [**UniProcessor: A Text-Induced Unified Low-Level Image Processor**](https://www.arxiv.org/abs/2407.20928) <br> | arXiv | 2024-07 | [Github](https://github.com/intmegroup/uniprocessor) |

### <a id="other-task">Other Task</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [**Large Language Models for Lossless Image Compression: Next-Pixel Prediction in Language Space is All You Need**](http://arxiv.org/abs/2411.12448) <br> | arXiv | 2024-11 | [-](-) |
| ![Star](https://img.shields.io/github/stars/kingnobro/Chat2SVG.svg?style=social&label=Star) <br> [**Chat2SVG: Vector Graphics Generation with Large Language Models and Image Diffusion Models**](http://arxiv.org/abs/2411.16602) <br> | arXiv | 2024-11 | [Github](https://github.com/kingnobro/Chat2SVG) |
| [**Image Regeneration: Evaluating Text-to-Image Model via Generating Identical Image with Multimodal Large Language Models**](http://arxiv.org/abs/2411.09449) <br> | arXiv | 2024-11 | [-](-) |
| ![Star](https://img.shields.io/github/stars/hyc2026/StoryTeller.svg?style=social&label=Star) <br> [**StoryTeller: Improving Long Video Description through Global Audio-Visual Character Identification**](http://arxiv.org/abs/2411.07076) <br> | arXiv | 2024-11 | [Github](https://github.com/hyc2026/StoryTeller) |
| [**EditScribe: Non-Visual Image Editing with Natural Language Verification Loops**](http://arxiv.org/abs/2408.06632) <br> | arXiv | 2024-08 | [-](-) |
| ![Star](https://img.shields.io/github/stars/RyannChenOO/MLeVLM.svg?style=social&label=Star) <br> [**MLeVLM: Improve Multi-level Progressive Capabilities based on Multimodal Large Language Model for Medical Visual Question Answering**](https://aclanthology.org/2024.findings-acl.296) <br> | Association for Computational Linguistics | 2024-08 | [Github](https://github.com/RyannChenOO/MLeVLM) |
| ![Star](https://img.shields.io/github/stars/templex98/mova.svg?style=social&label=Star) <br> [**MoVA: Adapting Mixture of Vision Experts to Multimodal Context**](http://arxiv.org/abs/2404.13046) <br> | arXiv | 2024-10 | [Github](https://github.com/templex98/mova) |
| ![Star](https://img.shields.io/github/stars/pku-yuangroup/chat-univi.svg?style=social&label=Star) <br> [**Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding**](http://arxiv.org/abs/2311.08046) <br> | arXiv | 2024-04 | [Github](https://github.com/pku-yuangroup/chat-univi) |
| ![Star](https://img.shields.io/github/stars/PKU-YuanGroup/Chat-UniVi.svg?style=social&label=Star) <br> [**Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding**](https://openaccess.thecvf.com/content/CVPR2024/html/Jin_Chat-UniVi_Unified_Visual_Representation_Empowers_Large_Language_Models_with_Image_CVPR_2024_paper.html) <br> | CVPR | 2024-06 | [Github](https://github.com/PKU-YuanGroup/Chat-UniVi) |
| ![Star](https://img.shields.io/github/stars/SkyworkAI/Vitron.svg?style=social&label=Star) <br> [**VITRON: A Unified Pixel-level Vision LLM for Understanding, Generating, Segmenting, Editing**](https://openreview.net/forum?id=kPmSfhCM5s) <br> | OpenReview | 2024 | [Github](https://vitron-llm.github.io/) |

### <a id="model-training">Model Training</a>

#### <a id="pre-training">Pre Training</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/HeimingX/TAG.svg?style=social&label=Star) <br> [**Attention-driven GUI Grounding: Leveraging Pretrained Multimodal Large Language Models without Fine-Tuning**](http://arxiv.org/abs/2412.10840) <br> | arXiv | 2024-12 | [Github](https://github.com/HeimingX/TAG) |
| [**From Visuals to Vocabulary: Establishing Equivalence Between Image and Text Token Through Autoregressive Pre-training in MLLMs**](http://arxiv.org/abs/2502.09093) <br> | arXiv | 2025-02 | [-](-) |
| ![Star](https://img.shields.io/github/stars/hanhuang22/AITQE.svg?style=social&label=Star) <br> [**Beyond Filtering: Adaptive Image-Text Quality Enhancement for MLLM Pretraining**](http://arxiv.org/abs/2410.16166) <br> | arXiv | 2024-10 | [Github](https://github.com/hanhuang22/AITQE) |
| ![Star](https://img.shields.io/github/stars/baaivision/Emu.svg?style=social&label=Star) <br> [**EMU: GENERATIVE PRETRAINING IN MULTIMODALITY**](https://openreview.net/forum?id=mL8Q9OOamV) <br> | OpenReview | 2024-03 | [Github](https://github.com/baaivision/Emu) |

#### <a id="fine-tuning">Fine Tuning</a>

|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/haotian-liu/LLaVA.svg?style=social&label=Star) <br> [**Visual Instruction Tuning**](http://arxiv.org/abs/2304.08485) <br> | arXiv | 2023-12 | [Github](https://github.com/haotian-liu/LLaVA) |
| ![Star](https://img.shields.io/github/stars/1xbq1/FedMLLM.svg?style=social&label=Star) <br> [**FedMLLM: Federated Fine-tuning MLLM on Multimodal Heterogeneity Data**](http://arxiv.org/abs/2411.14717) <br> | arXiv | 2024-11 | [Github](https://github.com/1xbq1/FedMLLM) |
| ![Star](https://img.shields.io/github/stars/PVIT-official/PVIT.svg?style=social&label=Star) <br> [**Position-Enhanced Visual Instruction Tuning for Multimodal Large Language Models**](http://arxiv.org/abs/2308.13437) <br> | arXiv | 2023-09 | [Github](https://github.com/PVIT-official/PVIT) |
| ![Star](https://img.shields.io/github/stars/DCDmllm/Cheetah.svg?style=social&label=Star) <br> [**Fine-tuning Multimodal LLMs to Follow Zero-shot Demonstrative Instructions**](http://arxiv.org/abs/2308.04152) <br> | arXiv | 2024-05 | [Github](https://github.com/DCDmllm/Cheetah) |
| ![Star](https://img.shields.io/github/stars/adaptNMT/adaptMLLM.svg?style=social&label=Star) <br> [**adaptMLLM: Fine-Tuning Multilingual Language Models on Low-Resource Languages with Integrated LLM Playgrounds**](http://arxiv.org/abs/2403.02370) <br> | - | 2023-11 | [Github](https://github.com/adaptNMT/adaptMLLM/) |
| ![Star](https://img.shields.io/github/stars/VT-NLP/MixLoRA.svg?style=social&label=Star) <br> [**Multimodal Instruction Tuning with Conditional Mixture of LoRA**](http://arxiv.org/abs/2402.15896) <br> | arXiv | 2024-12 | [Github](https://github.com/VT-NLP/MixLoRA) |
| [**Towards Robust Instruction Tuning on Multimodal Large Language Models**](http://arxiv.org/abs/2402.14492) <br> | arXiv | 2024-06 | [-](-) |
| [**Multi-modal Preference Alignment Remedies Degradation of Visual Instruction Tuning on Language Models**](http://arxiv.org/abs/2402.10884) <br> | - | 2024-02 | [-](-) |
| ![Star](https://img.shields.io/github/stars/jinlHe/PeFoMed.svg?style=social&label=Star) <br> [**PeFoMed: Parameter Efficient Fine-tuning of Multimodal Large Language Models for Medical Imaging**](http://arxiv.org/abs/2401.02797) <br> | arXiv | 2025-01 | [Github](https://github.com/jinlHe/PeFoMed) |
| ![Star](https://img.shields.io/github/stars/CircleRadon/Osprey.svg?style=social&label=Star) <br> [**Osprey: Pixel Understanding with Visual Instruction Tuning**](http://arxiv.org/abs/2312.10032) <br> | arXiv | 2024-03 | [Github](https://github.com/CircleRadon/Osprey) |
| [**CLAMP: Contrastive LAnguage Model Prompt-tuning**](http://arxiv.org/abs/2312.01629) <br> | arXiv | 2024-03 | [-](-) |
| ![Star](https://img.shields.io/github/stars/modelscope/ms-swift.svg?style=social&label=Star) <br> [**SWIFT:A Scalable lightWeight Infrastructure for Fine-Tuning**](http://arxiv.org/abs/2408.05517) <br> | arXiv | 2024-08 | [Github](https://github.com/modelscope/ms-swift) |
| ![Star](https://img.shields.io/github/stars/yunche0/GA-Net.svg?style=social&label=Star) <br> [**Multi-Modal Parameter-Efficient Fine-tuning via Graph Neural Network**](http://arxiv.org/abs/2408.00290) <br> | arXiv | 2024-08 | [Github](https://github.com/yunche0/GA-Net/tree/master) |
| ![Star](https://img.shields.io/github/stars/PhoenixZ810/MG-LLaVA.svg?style=social&label=Star) <br> [**MG-LLaVA: Towards Multi-Granularity Visual Instruction Tuning**](http://arxiv.org/abs/2406.17770) <br> | arXiv | 2024-06 | [Github](https://github.com/PhoenixZ810/MG-LLaVA) |
| ![Star](https://img.shields.io/github/stars/alenai97/PEFT-MLLM.svg?style=social&label=Star) <br> [**An Empirical Study on Parameter-Efficient Fine-Tuning for MultiModal Large Language Models**](http://arxiv.org/abs/2406.05130) <br> | arXiv | 2024-06 | [Github](https://github.com/alenai97/PEFT-MLLM) |
| ![Star](https://img.shields.io/github/stars/AIDC-AI/Parrot.svg?style=social&label=Star) <br> [**Parrot: Multilingual Visual Instruction Tuning**](http://arxiv.org/abs/2406.02539) <br> | arXiv | 2024-08 | [Github](https://github.com/AIDC-AI/Parrot) |
| [**Learn from Downstream and Be Yourself in Multimodal Large Language Model Fine-Tuning**](http://arxiv.org/abs/2411.10928) <br> | arXiv | 2024-11 | [-](-) |
| [**MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning**](http://arxiv.org/abs/2409.20566) <br> | arXiv | 2024-09 | [-](-) |
| [**Pilot: Building the Federated Multimodal Instruction Tuning Framework**](http://arxiv.org/abs/2501.13985) <br> | arXiv | 2025-01 | [-](-) |
| [**EACO: Enhancing Alignment in Multimodal LLMs via Critical Observation**](http://arxiv.org/abs/2412.04903) <br> | arXiv | 2024-12 | [-](-) |
| ![Star](https://img.shields.io/github/stars/-/-.svg?style=social&label=Star) <br> [**Visual Cue Enhancement and Dual Low-Rank Adaptation for Efficient Visual Instruction Fine-Tuning**](http://arxiv.org/abs/2411.12787) <br> | arXiv | 2024-12 | [-](-) |

<!-- #### <a id="optimization-and-enhancement">Optimization and Enhancement</a> -->

## <a id="related-surveys-recommended">Related Surveys Recommended</a>

**A Survey on Large Language Models for Recommendation**<br />
arXiv 2024. [[Paper](http://arxiv.org/abs/2305.19860)] <br />Jun. 2024<br />

**A Survey of Large Language Models**<br />
arXiv 2024. [[Paper](http://arxiv.org/abs/2303.18223)] <br />Oct. 2024<br />

**A Survey on Evaluation of Large Language Models**<br />
DOI (Crossref) 2024. [[Paper](https://dl.acm.org/doi/10.1145/3641289)] <br />Jun. 2024<br />

**MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs**<br />
arXiv 2024. [[Paper](http://arxiv.org/abs/2411.15296)] <br />Nov. 2024<br />

**Natural Language Understanding and Inference with MLLM in Visual Question Answering: A Survey**<br />
arXiv 2024. [[Paper](http://arxiv.org/abs/2411.17558)] <br />Nov. 2024<br />

**Large Language Model-Brained GUI Agents: A Survey**<br />
arXiv 2024. [[Paper](http://arxiv.org/abs/2411.18279)] <br />Nov. 2024<br />

**Visual Prompting in Multimodal Large Language Models: A Survey**<br />
arXiv 2024. [[Paper](http://arxiv.org/abs/2409.15310)] <br />Sep. 2024<br />

**A Survey of Camouflaged Object Detection and Beyond**<br />
arXiv 2024. [[Paper](http://arxiv.org/abs/2408.14562)] <br />Aug. 2024<br />

**MM-LLMs: Recent Advances in MultiModal Large Language Models**<br />
arXiv 2024. [[Paper](http://arxiv.org/abs/2401.13601)] <br />May. 2024<br />

**A Survey on Benchmarks of Multimodal Large Language Models**<br />
arXiv 2024. [[Paper](http://arxiv.org/abs/2408.08632)] <br />Sep. 2024<br />

**A Survey on Multimodal Benchmarks: In the Era of Large AI Models**<br />
arXiv 2024. [[Paper](http://arxiv.org/abs/2409.18142)] <br />Sep. 2024<br />

**Multimodal Image Synthesis and Editing: The Generative AI Era**<br />
arXiv 2023. [[Paper](http://arxiv.org/abs/2112.13592)] <br />Aug. 2023<br />

**A Survey on Visual Transformer**<br />
arXiv 2023. [[Paper](http://arxiv.org/abs/2012.12556)] <br />Jul. 2023<br />

**Personalized Multimodal Large Language Models: A Survey**<br />
arXiv 2024. [[Paper](http://arxiv.org/abs/2412.02142)] <br />Dec. 2024<br />

**Explainable and Interpretable Multimodal Large Language Models: A Comprehensive Survey**<br />
arXiv 2024. [[Paper](http://arxiv.org/abs/2412.02104)] <br />Dec. 2024<br />

**A Survey on Vision-Language-Action Models for Embodied AI**<br />
arXiv 2024. [[Paper](http://arxiv.org/abs/2405.14093)] <br />Nov. 2024<br />

**Multimodal Learning With Transformers: A Survey**<br />
IEEE 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/10123038)] <br />2023<br />

**A Survey of Multimodal Large Language Model from A Data-centric Perspective**<br />
arXiv 2024. [[Paper](http://arxiv.org/abs/2405.16640)] <br />Dec. 2024<br />

**A Survey on All-in-One Image Restoration: Taxonomy, Evaluation and Future Trends**<br />
arXiv 2024. [[Paper](http://arxiv.org/abs/2410.15067)] <br />Oct. 2024<br />

**Vision Transformers in Image Restoration: A Survey**<br />
mdpi 2023. [[Paper](https://www.mdpi.com/1424-8220/23/5/2385)] <br />Jan. 2023<br />

**Recent progress in image denoising: A training strategy perspective**<br />
 [[Paper](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.12748)] <br />May. 2023<br />

**Survey of different Large Language Model Architectures: Trends, Benchmarks, and Challenges**<br />
arXiv 2024. [[Paper](http://arxiv.org/abs/2412.03220)] <br />Dec. 2024<br />

**Survey of Large Multimodal Model Datasets, Application Categories and Taxonomy**<br />
arXiv 2025. [[Paper](http://arxiv.org/abs/2412.17759)] <br />Feb. 2025<br />

**Image, Text, and Speech Data Augmentation using Multimodal LLMs for Deep Learning: A Survey**<br />
arXiv 2025. [[Paper](http://arxiv.org/abs/2501.18648)] <br />Jan. 2025<br />

**How to Bridge the Gap between Modalities: A Comprehensive Survey on Multi-modal Large Language Model**<br />
arXiv 2023. [[Paper](https://arxiv.org/abs/2311.07594)] <br />Nov. 2023<br />

**Multimodal Large Language Models: A Survey**<br />
IEEE 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/10386743)] <br />2023<br />

**Vision-Language Models for Vision Tasks: A Survey**<br />
IEEE 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/10445007)] [[Github](https://github.com/jingyi0000/vlm_survey)] <br />2023<br />

**From Word Vectors to Multimodal Embeddings: Techniques, Applications, and Future Directions For Large Language Models**<br />
arXiv 2024. [[Paper](http://arxiv.org/abs/2411.05036)] <br />Nov. 2024<br />

<!-- ## <a id="awesome-datasets">Awesome Datasets</a>

### <a id="dehazing">Dehazing</a>

### <a id="deblurring">Deblurring</a>

### <a id="deraining">Deraining</a>

### <a id="desnowing">Desnowing</a>

### <a id="denoising">Denoising</a>

### <a id="other">Other</a> -->

## <a id="benchmarks-for-evaluation">Benchmarks for Evaluation</a>

|  Name  |   Paper  |   Link   |   Notes   |
|:--------|:--------:|:--------:|:--------:|
| **Q-Bench** | [Q-Bench: A Benchmark for General-Purpose Foundation Models on Low-level Vision](http://arxiv.org/abs/2309.14181) | [repo](https://q-future.github.io/Q-Bench) | A holistic benchmark crafted to systematically evaluate potential abilities of MLLMs on three realms: low-level visual perception, low-level visual description, and overall visual quality assessment. |
| **Q-Bench<sup>+</sup>** | [Q-Bench<sup>+</sup>: A Benchmark for Multi-modal Foundation Models on Low-level Vision from Single Images to Pairs](http://arxiv.org/abs/2402.07116) | [repo](https://github.com/Q-Future/Q-Bench) | A benchmark settings to emulate human language responses related to low-level vision: the low-level visual perception via visual question answering related to low-level attributes; and the low-level visual description, on evaluating MLLMs for low-level text descriptions. |
| **HEIE** | [HEIE: MLLM-Based Hierarchical Explainable AIGC Image Implausibility Evaluator](http://arxiv.org/abs/2411.17261) | [repo](-) | A novel MLLM-Based Hierarchical Explainable image Implausibility Evaluator. |
| **QL-Bench** | [Explore the Hallucination on Low-level Perception for MLLMs](http://arxiv.org/abs/2409.09748) | [repo](-) | - |
| **MLLM-as-a-Judge** | [MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark](https://arxiv.org/abs/2402.04788) | [repo](https://github.com/Dongping-Chen/MLLM-Judge) | - |
| **Q-BOOST** | [Q-BOOST: On Visual Quality Assessment Ability of Low-Level Multi-Modality Foundation Models](https://ieeexplore.ieee.org/abstract/document/10645451) | [repo](https://github.com/Q-Future/Q-Instruct/boost_qa) | A focused exploration of the visual quality assessment capabilities in low-level multi-modality foundation models, introducing Q-BOOST as a benchmark framework. |
| **SEED-Bench** | [SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension](https://arxiv.org/abs/2307.16125) | [repo](https://github.com/AILab-CVC/SEED-Bench) | A benchmark designed to evaluate the generative comprehension abilities of multimodal large language models across diverse tasks and datasets. |
| **MIBench** | [MIBench: Evaluating Multimodal Large Language Models over Multiple Images](http://arxiv.org/abs/2407.15272) | [repo](-) | - |
| **AICoderEval** | [AICoderEval: Improving AI Domain Code Generation of Large Language Models](http://arxiv.org/abs/2406.04712) | [repo](-) | - |
| **MLe-Bench** | [MLeVLM: Improve Multi-level Progressive Capabilities based on Multimodal Large Language Model for Medical Visual Question Answering](https://aclanthology.org/2024.findings-acl.296) | [repo](-) | - |

### <a id="metrics">Metrics</a>

- **Image Quality Assessment (IQA)**
  - **PSNR (Peak Signal-to-Noise Ratio)** <br />
PSNR is a metric based on pixel differences, measuring the Mean Squared Error (MSE) between the generated and real images. A higher PSNR value indicates better image quality.
  - **SSIM (Structural Similarity Index Measure)** <br />
SSIM is a metric based on the structural similarity of images, taking into account the similarities in luminance, contrast, and structure. An SSIM value closer to 1 indicates better image quality.
  - **LPIPS (Learned Perceptual Image Patch Similarity)** <br />
LPIPS is a perceptual similarity metric based on deep learning, which measures the perceptual difference between images by training convolutional neural networks. A smaller LPIPS value indicates better image quality.
- **Perceptual Quality Assessment (PQA)**
  - **NIQE (Naturalness Image Quality Evaluator)** <br />
NIQE is a no-reference perceptual quality assessment metric that evaluates the realism of an image by analyzing its natural statistical characteristics. A lower NIQE value indicates that the image is closer to natural images.
  - **BLIND (Blind/Referenceless Image Spatial Quality Evaluator)** <br />
BLIND is a no-reference perceptual quality assessment metric that evaluates image quality by analyzing its spatial features. A lower BLIND/BIQ value indicates better image quality.
  - **FR-IQA (Full-Reference Image Quality Assessment)** <br />
FR-IQA is a full-reference perceptual quality assessment metric that evaluates the quality of generated images by comparing their perceptual differences with real images. Common FR-IQA metrics include LPIPS, MS-SSIM, etc.
- **Structural Similarity Assessment (SSA)**
  - **MS-SSIM (Multi-Scale Structural Similarity Index Measure)** <br />
MS-SSIM is an extended version of SSIM that evaluates overall image quality by calculating structural similarity across multiple scales. An MS-SSIM value closer to 1 indicates better image quality.
  - **VIF (Visual Information Fidelity)** <br />
VIF is an information-theoretic metric that assesses image quality by measuring the visual information fidelity between the generated image and the real image. A higher VIF value indicates better image quality.

## <a id="reference">Reference</a>
[Awesome-Multimodal-Large-Language-Models-by-BradyFU](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)

