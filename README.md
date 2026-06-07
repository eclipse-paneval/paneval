<p align="center">
  <img src="assets/logo.png" width="400" alt="PanEval Logo">
</p>

<p align="center">
  <strong>A Flexible Framework for Comprehensive Multimodal Model Evaluation</strong>
</p>

<p align="center">
  <a href="https://github.com/eclipse-paneval/paneval/actions"><img src="https://github.com/eclipse-paneval/paneval/actions/workflows/ci.yaml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/paneval/"><img src="https://img.shields.io/pypi/v/paneval" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-%3E%3D3.10-blue" alt="Python"></a>
  <a href="https://github.com/eclipse-paneval/paneval/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://paneval.readthedocs.io/"><img src="https://img.shields.io/badge/docs-latest-brightgreen" alt="Docs"></a>
</p>

<p align="center">
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-supported-tasks">Tasks</a> •
  <a href="https://paneval.readthedocs.io/">Docs</a>
</p>

---

## 📖 Overview

PanEval is an open-source evaluation framework for comprehensive assessment of multimodal models. It provides a standardized, extensible platform to evaluate models across diverse modalities — text, images, and video — with a rich set of benchmarks and metrics.

Whether you are benchmarking state-of-the-art vision-language models, evaluating text-to-image generation quality, or measuring video retrieval accuracy, PanEval offers a unified interface with reproducible configurations.

## ✨ Key Features

- **🎯 Unified Evaluation Interface** — A single CLI command to evaluate any model across multiple benchmarks, with consistent output formats and caching.
- **📊 40+ Built-in Benchmarks** — Covers VQA, image/video retrieval, text-to-image/video generation, math reasoning, spatial understanding, chart interpretation, and more.
- **🧩 Pluggable Model Adapters** — Bring your own model via a simple adapter interface. Built-in support for vLLM, SGLang, LMDeploy, Transformers, and HTTP APIs.
- **🤖 LLM-as-Judge** — Leverage GPT, Claude, Gemini, and other LLMs as automated evaluators for open-ended tasks.
- **⚙️ Reproducible Configs** — JSON-based model configurations ensure consistent, version-controlled evaluation pipelines.
- **🔌 Extensible Architecture** — Easily add new datasets, evaluators, and model adapters via a registry-based plugin system.

## 🚀 Installation

```bash
pip install paneval
```

For development:

```bash
git clone https://github.com/eclipse-paneval/paneval.git
cd paneval
pip install -e .
```

### Optional Backends

| Backend | Install Command |
|---------|----------------|
| vLLM | `pip install vllm` |
| SGLang | `pip install "sglang[all]"` |
| LMDeploy | `pip install lmdeploy` |
| Flash-Attention | `pip install flash-attn --no-build-isolation` |

### API Key Setup

To use GPT or Claude as evaluators, set the following environment variables:

```bash
export EVALMM_API_KEY="your-api-key"
export EVALMM_BASE_URL="https://api.openai.com/v1"
```

## ⚡ Quick Start

Evaluate a model on MMMU using vLLM:

```bash
evalmm --tasks tasks/mmmu/mmmu_val.py \
       --exec model_zoo/vlm/api_model/model_adapter.py \
       --model llava-hf/llava-onevision-qwen2-7b-ov-chat-hf \
       --backend vllm \
       --num-workers 8 \
       --output-dir ./results/llava-ov-7b
```

Using a JSON config:

```bash
evalmm --cfg model_configs/open/Qwen2.5-VL-7B-Instruct.json \
       --tasks tasks/mmmu/mmmu_val.py \
       --exec model_zoo/vlm/api_model/model_adapter.py
```

Evaluate with GPT-4o as judge:

```bash
evalmm --tasks tasks/blink/blink_val.py \
       --exec model_zoo/vlm/api_model/model_adapter.py \
       --model gpt-4o-mini \
       --num-workers 4 \
       --api-key $OPENAI_API_KEY \
       --url https://api.openai.com/v1/chat/completions \
       --output-dir ./results/gpt-4o-mini
```

## 📚 Usage

### Single Task

```bash
evalmm --tasks tasks/chart_qa/chart_qa_test.py \
       --cfg model_configs/open/InternVL2-8B.json \
       --exec model_zoo/vlm/api_model/model_adapter.py \
       --backend vllm \
       --output-dir ./results/internvl2-8b
```

### Multiple Tasks

```bash
evalmm --tasks tasks/mmmu/mmmu_val.py \
              tasks/mmvet/mmvet_v2.py \
              tasks/ocrbench/ocrbench_test.py \
       --cfg model_configs/open/Qwen2.5-VL-72B-Instruct.json \
       --exec model_zoo/vlm/api_model/model_adapter.py \
       --backend vllm \
       --output-dir ./results/qwen2.5-vl-72b
```

### Multi-Inference (Temperature > 0)

```bash
evalmm --cfg model_configs/open/Qwen2.5-VL-7B-Instruct.json \
       --tasks tasks/blink/blink_val.py \
       --exec model_zoo/vlm/api_model/model_adapter.py \
       --backend vllm \
       --num-infers 5 \
       --temperature 0.6 \
       --output-dir ./results/qwen2.5-vl-7b-multi
```

### Separate Data Server & Evaluation

```bash
# Terminal 1: Start the data server
python evalmm/server/run_server.py \
    --tasks tasks/charxiv/charxiv_val.py \
    --output-dir ./results/qwen2-vl-7b \
    --port 11823

# Terminal 2: Run evaluation
python evalmm/eval.py \
    --output-dir ./results/qwen2-vl-7b \
    --tasks tasks/charxiv/charxiv_val.py \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --exec model_zoo/vlm/qwen_vl/model_adapter.py \
    --server-port 11823
```

## 📊 Supported Tasks

PanEval includes **40+ built-in evaluation tasks** covering a wide range of multimodal capabilities:

| Category | Tasks |
|----------|-------|
| **General VQA** | MMMU, MMMU-Pro, MM-Vet, CMMMU, CMMU, MLVU, CV-Bench, RealWorldQA |
| **Document & Text** | OCRBench, TextVQA, CharXiv, ChartQA, CII-Bench |
| **Math & Reasoning** | MathVista, MathVerse, MathVision, MeasureBench |
| **Spatial & Embodied** | SpatialBench, EmbSpatialBench, OmniSpatial, RoboSpatialHome, Where2Place |
| **Image Retrieval** | COCO, Flickr, MSR-VTT, UCF-101 |
| **T2I Generation** | COCO-T2I, GenAI-Bench, RelScene |
| **T2V Generation** | UCF-101 (video), SoraPrompt |
| **Safety** | VideoSafetyBench |
| **Specialized** | BLINK, ARPGrounding, ROME, ERQA, HallusionBench, AnimalBench, SatBench, VSI-Bench, TRUe, VisualSimpleQA |

## 🏗 Architecture

```
paneval/
├── evalmm/              # Core framework
│   ├── dataset/         # Dataset loading & preprocessing
│   ├── evaluator/       # Evaluation engines (extract, retrieval, MMMU, etc.)
│   ├── models/          # Model clients (GPT, Claude, Gemini, Hunyuan, etc.)
│   ├── prompt/          # Prompt templates
│   ├── server/          # Data server & evaluation server
│   └── common/          # Utilities, constants, logging
├── tasks/               # 40+ benchmark task definitions
├── model_zoo/           # Model adapters (VLM, T2I, T2V, Retrieval)
├── model_configs/       # JSON model configuration files
├── tools/               # CLI tools and batch runner
└── skills/              # Skill modules for dataset integration
```

## 🤝 Contributing

We welcome contributions! To add a new dataset or benchmark:

1. Follow the [Adding a Task](tasks/ADD_TASK_ZH.md) guide
2. Use the `skills/evalmm-add-dataset` skill for structured integration
3. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📝 Citation

If you use PanEval in your research, please cite:

```bibtex
@inproceedings{he-etal-2025-flagevalmm,
    title = "FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation",
    author = "He, Zheqi and Liu, Yesheng and Zheng, Jing-Shu and Li, Xuejing and Yao, Jin-Ge and Qin, Bowen and Xuan, Richeng and Yang, Xi",
    editor = "Mishra, Pushkar and Muresan, Smaranda and Yu, Tao",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-demo.6/",
    pages = "51--61",
    ISBN = "979-8-89176-253-4"
}
```

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
