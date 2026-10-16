<p align="center">
  <img src="assets/logo.svg" alt="MLEvolve" width="400"/>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.06473"><img src="https://img.shields.io/badge/arXiv-2606.06473-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/papers/2606.06473"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow.svg" alt="Hugging Face"></a>
  <a href="https://internscience.github.io/MLEvolve/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://github.com/InternScience/MLEvolve"><img src="https://img.shields.io/github/stars/InternScience/MLEvolve?style=social" alt="GitHub Stars"></a>
</p>

An agentic MLE (Machine Learning Engineering) system that automatically solves Kaggle-style ML competitions through Monte Carlo Graph Search (MCGS) with multi-agent collaboration. This is an advanced version based on [AutoMLGen](https://arxiv.org/abs/2510.08511). MLEvolve achieves **#1 on the [MLE-bench](https://github.com/openai/mle-bench) leaderboard** (65.3% medal rate, 12-hour budget) and obtains competitive results on mathematical optimization tasks.

## Timeline

- **2026-10-15** — 🔥 We release **MLExpert**, a 35B MoE agentic model for autonomous MLE, achieving **72.7%** medal rate on MLE-Bench Lite. [[Paper](https://arxiv.org/abs/2610.22344)] [[Code](https://github.com/InternScience/MLExpert)] [[Model](https://huggingface.co/InternScience/MLExpert)]
- **2026-06-05** — Paper released on arXiv: [MLEvolve: A Self-Evolving Framework for Automated Machine Learning Algorithm Discovery](https://arxiv.org/abs/2606.06473).
- **2026-06-01** — **65.3% medal rate** on full MLE-bench (75 tasks), also achieves competitive results on mathematical optimization tasks.
- **2026-03-23** — Now supports OpenAI-compatible APIs (GPT, Qwen, DeepSeek, etc.). Models with function calling support are recommended for best performance.
- **2026-02-14** — MLEvolve codebase is now open-source.
- **2026-02-14** — MLEvolve achieves **#1 on MLE-bench** (12-hour budget).


## MLE-bench Results

Performance on the [MLE-bench](https://github.com/openai/mle-bench) full set (75 tasks). Medal rates reported across three complexity levels and overall (mean ± SEM over 3 seeds).

#### Proprietary Methods

| Agent | LLM | Time (h) | Low (%) | Medium (%) | High (%) | All (%) |
|-------|-----|:---------:|---------|------------|----------|---------|
| FM-Agent | Gemini-2.5-Pro | 24 | 62.1 ± 1.5 | 36.8 ± 1.5 | 33.3 ± 0.0 | 43.6 ± 0.9 |
| MLE-STAR-Pro-1.5 | Gemini-2.5-Pro | 24 | 68.2 ± 2.6 | 34.2 ± 1.5 | 33.3 ± 0.0 | 44.0 ± 1.3 |
| MARS | Gemini-3-Pro-preview | 24 | 74.2 ± 1.5 | 52.6 ± 3.0 | 37.8 ± 2.2 | 56.0 ± 1.5 |
| MARS+ | Gemini-3-Pro-preview | 24 | 78.8 ± 1.5 | 60.5 ± 1.5 | 44.4 ± 2.2 | 62.7 ± 0.8 |
| AIBuildAI | Claude-Opus-4.6 | 24 | 77.3 ± 0.0 | 61.4 ± 0.9 | 46.7 ± 0.0 | 63.1 ± 0.4 |

#### Open-Source Methods

| Agent | LLM | Time (h) | Low (%) | Medium (%) | High (%) | All (%) |
|-------|-----|:---------:|---------|------------|----------|---------|
| AIDE | o1-preview | 24 | 35.9 ± 1.9 | 8.5 ± 0.4 | 11.7 ± 1.3 | 17.1 ± 0.6 |
| ML-Master | DeepSeek-R1 | 12 | 48.5 ± 1.5 | 20.2 ± 2.3 | 24.4 ± 2.2 | 29.3 ± 0.8 |
| AIRA-Dojo | o3 | 24 | 55.0 ± 1.5 | 22.0 ± 1.2 | 21.7 ± 1.1 | 31.6 ± 0.8 |
| R&D-Agent | gpt-5 | 12 | 68.2 ± 2.6 | 21.1 ± 1.5 | 22.2 ± 2.2 | 35.1 ± 0.4 |
| Leeroo | Gemini-3-Pro-preview | 24 | 68.2 ± 2.6 | 44.7 ± 1.5 | 40.0 ± 0.0 | 50.7 ± 1.3 |
| ML-Master 2.0 | DeepSeek-V3.2-Speciale | 24 | 75.8 ± 1.5 | 50.9 ± 3.5 | 42.2 ± 2.2 | 56.4 ± 2.5 |
| **MLEvolve (Ours)** | **Gemini-3.1-Pro-preview** | **12** | **80.3 ± 1.5** | **64.0 ± 0.9** | **46.7 ± 0.0** | **65.3 ± 0.8** |


## Mathematical Optimization Results

Beyond machine-learning engineering, MLEvolve **generalizes** to open problems in mathematical optimization. Every baseline here (AlphaEvolve and its v2, SimpleTES, TTT-Discover, OpenEvolve) is a framework **purpose-built for optimization**; even so, MLEvolve **matches or surpasses AlphaEvolve on 14 of 15 tasks** and achieves the **best result among all compared methods on 11 of 15 tasks**.

Comparison on 15 mathematical programming tasks. Each column header carries the optimization direction (**↑** = higher is better, **↓** = lower is better); **bold** marks the best value. The 15 tasks are split across three tables so the page fits — see the legend below for full task names.

**Tasks 1–5**

| Method | Hex packing ↓ | Circle/square ↑ | Circle/rect ↑ | Heilbronn convex ↑ | Heilbronn tri ↑ |
|--------|---------------|-----------------|---------------|--------------------|-----------------|
| AlphaEvolve | 3.930092 | 2.6358627564 | 2.3658321334 | 0.0309368890 | 0.03652988988003016 |
| AlphaEvolve-v2 | 3.931 | — | — | 0.0309 | 0.0365 |
| SimpleTES | 3.931 | 2.635983 | — | — | — |
| TTT-Discover | — | — | — | — | — |
| OpenEvolve | — | — | — | — | — |
| **MLEvolve** | **3.9284759302** | **2.6359830395** | **2.3658323759** | **0.0309372079** | **0.03652988988003020** |

**Tasks 6–10**

| Method | Kissing d11 ↑ | Sum-diff 1 ↑ | Sum-diff 2 ↑ | Uncertainty ↓ | Autocorr 1 ↓ |
|--------|---------------|--------------|--------------|---------------|--------------|
| AlphaEvolve | **593** | 1.1479889651 | 1.1584172816 | 0.3520991044225 | 1.5052939684 |
| AlphaEvolve-v2 | **593** | 1.1479 | 1.1584 | 0.3521 | 1.5032 |
| SimpleTES | — | 1.143975 | — | — | 1.503871 |
| TTT-Discover | — | — | — | — | 1.5028628983 |
| OpenEvolve | — | — | — | — | 1.507190 |
| **MLEvolve** | 592 | **1.1901774219** | **1.1585457700** | **0.3520991044160** | **1.5028628749** |

**Tasks 11–15**

| Method | Autocorr 3v ↓ | Autocorr 3 ↓ | Autocorr 2 ↑ | Max/min ↓ | Min overlap ↓ |
|--------|---------------|--------------|--------------|-----------|---------------|
| AlphaEvolve | 1.4687620697 | 1.4556427954 | 0.8962799442 | 12.88926611203 | 0.3809230351 |
| AlphaEvolve-v2 | — | 1.4557 | 0.961 | 12.889266112 | 0.380924 |
| SimpleTES | — | **1.453675** | **0.962694** | — | **0.380868** |
| TTT-Discover | — | — | 0.959100 | — | 0.3808753232 |
| OpenEvolve | — | 1.460000 | 0.944900 | — | 0.380965 |
| **MLEvolve** | **1.4587698922** | 1.4548507482 | 0.9054217971 | **12.8892299077** | 0.3808968496 |

<details>
<summary><b>Task legend</b> (short header → full task name)</summary>

| Short | Full task name |
|-------|----------------|
| Hex packing | Packing hexagons in hexagons |
| Circle/square | Circle packing in a square |
| Circle/rect | Circle packing in a rectangle |
| Heilbronn convex | Heilbronn convex regions |
| Heilbronn tri | Heilbronn triangles |
| Kissing d11 | Kissing number dim 11 |
| Sum-diff 1 | Sums differences problems 1 |
| Sum-diff 2 | Sums differences problems 2 |
| Uncertainty | An uncertainty inequality |
| Autocorr 1 | First autocorrelation inequality |
| Autocorr 3v | Third autocorrelation (variant) |
| Autocorr 3 | Third autocorrelation inequality |
| Autocorr 2 | Second autocorrelation inequality |
| Max/min | Max-to-min ratios |
| Min overlap | Minimum Overlap Problem |

</details>


## MLExpert: Agentic Model for MLE

We further release **[MLExpert](https://arxiv.org/abs/2610.22344)**, a 35B MoE agentic model post-trained for long-horizon autonomous MLE via hierarchical agentic SFT and Windowed Interaction RL (WIRL). It raises the MLE-Bench Lite medal rate from 37.9% to **72.7%**, comparable to much larger frontier models. See the [code](https://github.com/InternScience/MLExpert) and [model weights](https://huggingface.co/InternScience/MLExpert).


## Coding Module in AI-Scientist

MLEvolve powers the **coding and algorithm optimization** module within the [InternAgent](https://github.com/InternScience/InternAgent) system. Built on MLEvolve's refinement engine, [InternAgent 1.5](https://arxiv.org/abs/2602.08990) **further enables autonomous algorithm design and end-to-end scientific discovery**.


## Key Technical Contributions

**Multi-Mode Planning & Code Generation** — Supports base (single-shot) and memory-enhanced (two-stage retrieval-augmented) planning, paired with three code generation strategies: single-pass, stepwise multi-agent pipeline, and incremental SEARCH/REPLACE diff patching. Different modes are dispatched adaptively based on search state.

**Experience-Driven Memory** — A global memory layer records plan, code, metrics, and success/failure labels for every node. Retrieval combines BM25 + FAISS allowing the planner to reinforce proven strategies and avoid known pitfalls from its own search history. Different agents query memory in different ways to encourage novel approaches.

**Progressive MCGS with Cross-Branch Fusion** — The search graph extends vanilla UCT with piecewise exploration decay, time-aware explore-exploit switching, and automatic stagnation detection. Multiple solution branches evolve in parallel; when progress stalls, the system performs cross-branch fusion — merging insights from top-performing nodes across different branches into new solution candidates — and trajectory-aware evolution that leverages each branch's full improvement history to propose informed next steps.



## Setup

**1. Prepare mle-bench** — Install [mle-bench](https://github.com/openai/mle-bench) and download the dataset following its instructions.

**2. Install MLEvolve dependencies**

```bash
pip install --no-deps -r requirements_base.txt
pip install --no-deps -r requirements_ml.txt
pip install --no-deps -r requirements_domain.txt
```

**3. Configure** — Edit `config/config.yaml`, fields you **must** fill in:

```yaml
dataset_dir: "/path/to/mle-bench/data"

agent:
  code:
    base_url: "https://your-gemini-endpoint"
    api_key: "your-api-key"
  feedback:
    base_url: "https://your-gemini-endpoint"
    api_key: "your-api-key"
```

Other tunable fields (`agent.steps`, `agent.time_limit`, etc.) have sensible defaults — see comments in the yaml file.

If `agent.use_global_memory: True`, you must also set `agent.memory_embedding_model_path` to a valid HuggingFace embedding model name or local model path. Set `agent.memory_embedding_device` to `cpu` if CUDA is unavailable.

### Cold-Start Models (optional)

Cold-start recommends pretrained models per task category based on `engine/coldstart/models_guidance_classified.json`. Most models auto-download from HuggingFace; for models requiring local weights, set `torch_hub_dir` in `config.yaml`. To disable cold-start entirely, set `coldstart.use_coldstart: False`.

## Quick Start

```bash
bash run_single_task.sh <EXP_ID> <DATASET_DIR> [SERVER_ID]

# Example
bash run_single_task.sh denoising-dirty-documents /mle-bench/data 1
```

Results are written to `./runs/<timestamp>_<exp_id>/` including search tree logs, best solution code, and top-K candidate submissions.

## Acknowledgments

We thank [AIDE](https://github.com/WecoAI/aideml) and [ML-Master](https://github.com/sjtu-sai-agents/ML-Master) for their contributions to the development of the MCTS in MLE, and [InternAgent 1.5](https://github.com/InternScience/InternAgent) for its contributions to the development of the agentic memory mechanism. We sincerely thank all teams for their open-source contributions to the community.

## Citation

If you find this repo useful, please cite our work.

```bibtex
@article{du2026mlevolve,
  title={MLEvolve: A Self-Evolving Framework for Automated Machine Learning Algorithm Discovery},
  author={Du, Shangheng and Yan, Xiangchao and Shi, Jinxin and Cao, Zongsheng and Feng, Shiyang and Liang, Zichen and Sun, Boyuan and Peng, Tianshuo and Zhou, Yifan and Li, Xin and Zhou, Jie and He, Liang and Zhang, Bo and Bai, Lei},
  journal={arXiv preprint arXiv:2606.06473},
  year={2026}
}
```

Early version:
```bibtex
@article{du2025automlgen,
  title={AutoMLGen: Navigating Fine-Grained Optimization for Coding Agents},
  author={Du, Shangheng and Yan, Xiangchao and Jiang, Dengyang and Yuan, Jiakang and Hu, Yusong and Li, Xin and He, Liang and Zhang, Bo and Bai, Lei},
  journal={arXiv preprint arXiv:2510.08511},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
