# ManyICLBench

> 📄 Accepted to **ACL 2025 (Main Conference)**  
> 🔗 [Paper](https://arxiv.org/abs/2411.07130) | 🏆 [Leaderboard](https://huggingface.co/spaces/launch/ManyICLBench_Leaderboard) | 📊 [Dataset](https://huggingface.co/datasets/launch/ManyICLBench)

---

## 📌 Overview

**ManyICLBench** is a benchmark designed to evaluate long-context language models (LCLMs) via **many-shot in-context learning (ICL)**. We investigate whether performance improves with additional demonstrations and introduce a new metric, **Sample Learning Ratio (SLR)**, to characterize task types:

- **SSL (Similar-Sample Learning)**: Tasks where models benefit from retriving similar demostrations.
- **ASL (All-Sample Learning)**: Tasks where models need to understand all demonstrations.

---
## ⚙️ Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run evaluation on a model

```bash
bash start_vllm_serve.sh
```
Wait till the server starts
```bash
bash evaluate.sh
```

### 3. Generate your final results

```bash
python create_csv.py
```
---
## 🚀 Leaderboard

Submit your model results at:  
📍 https://huggingface.co/spaces/launch/ManyICLBench_Leaderboard

## 📑 Citation

If you use our benchmark or results in your work, please cite us:

```bibtex
@article{zou2025manyshotincontextlearninglongcontext,
  title={On Many-Shot In-Context Learning for Long-Context Evaluation}, 
  author={Kaijian Zou and Muhammad Khalifa and Lu Wang},
  journal={arXiv preprint arXiv:2411.07130},
  year={2025}
}
```
---

## 📬 Contact

- 🧑‍💻 Lead author: [Kaijian Zou] ([zkjzou@umich.edu])
- ❓ For questions or bugs: please open an [issue](https://github.com/launchnlp/ManyICLBench/issues)

## 🙏 Acknowledgements
Part of the codebase is based on [RULER](https://github.com/NVIDIA/RULER/tree/main)

We thank the reviewers at ICLR 2025 and ACL 2025 for their insightful feedback. We also appreciate the Hugging Face and vLLM communities for their tools and infrastructure, which greatly supported this project.