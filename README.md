# ChatEvac

**End-to-End Automation Assessment for Building Evacuation Safety Based on Large Language and Diffusion Models**

ChatEvac is an intelligent agent powered by a Large Language Model (LLM) that automates the full pipeline of evacuation safety assessment — from a user-uploaded architectural floorplan to a regulatory-grounded evaluation report. A Diffusion model segments spatial features, a Social Force Model (SFM) engine executes pedestrian simulations, and an LLM orchestrates the workflow through an interactive GUI. Three complementary mechanisms — a Finite State Machine (FSM), Retrieval-Augmented Generation (RAG), and a human-in-the-loop verification checkpoint — work together to mitigate LLM hallucination in this safety-sensitive context.

<p align="center">
  <img src="img/Framework%20and%20functional%20layers%20of%20ChatEvac.png" alt="Framework and functional layers of ChatEvac" width="80%">
</p>

## Architecture Overview

ChatEvac is built around a **hybrid AI framework** centered on an LLM pivot (GPT-4o). Four functional layers are orchestrated through a 15-state Finite State Machine:

| Layer | Module | Function |
|-------|--------|----------|
| **Input & GUI** | `Agent.py` (Tkinter) | Interactive dialog, floorplan upload, human-in-the-loop parameter verification |
| **Feature Segmentation** | ControlNet + Stable Diffusion v1.5 | Extracts walls (white), indoor areas (black), and exits (red) from floorplans |
| **Evacuation Simulation** | SFM engine + Voronoi navigation mesh | Physics-based pedestrian dynamics with Dijkstra path planning |
| **Analysis & Report** | RAG + GPT-4o | Regulatory-grounded evaluation report with heatmap visualizations |

<p align="center">
  <img src="img/LLM%20orchestration%20framework%20and%20GUI%20interaction%20flow.png" alt="LLM orchestration framework and GUI interaction flow" width="90%">
</p>

### Three-Layer Hallucination Defense Architecture

LLM hallucination is a critical risk in safety engineering. ChatEvac employs a **multi-layer defense**:

1. **FSM (Workflow-level)** — A 15-state finite state machine constrains the LLM to valid workflow transitions. The LLM can only output symbols among legally permitted actions for the current state, improving workflow adherence by **21.4%** over an unconstrained baseline.

2. **RAG (Content-level)** — 518 building code provisions from NFPA 101 and IBC are embedded and indexed via FAISS. Retrieved provisions are injected into the system prompt, improving citation accuracy by **40.4%**.

3. **Human-in-the-Loop (Operational-level)** — LLM-extracted simulation parameters are displayed for explicit user confirmation before computationally intensive simulation runs, intercepting residual errors at module boundaries.


## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for ControlNet inference and training)
- Git

### 1. Clone and set up environment

```bash
git clone https://github.com/your-username/ChatEvac.git
cd ChatEvac
pip install -r requirements.txt
```

### 2. (Optional) Build the RAG index

The RAG module requires a pre-built FAISS index for building code retrieval. A pre-built index is included; to rebuild it from source provisions:

```bash
cd RAG
python build_index.py
```

> **Note:** `build_index.py` requires the API key configured in `config.py` to call the OpenAI embeddings API.

### 3. Configure API keys

Edit `config.py` in the project root and fill in your API keys:

```python
CHAT_API_KEY = "your-openai-api-key-here"
CHAT_API_BASE = "https://api.openai.com/v1"   # or compatible endpoint
```

See `config.py` for all available options. Each configuration block documents which scripts use it and which capabilities are required.

## Usage

### Step 1 — Train the Diffusion Model (optional)

If you have your own floorplan dataset:

```bash
# 1. Prepare data under dataset/source/ and dataset/target/
# 2. Create dataset/prompt.jsonl with metadata
# 3. Edit train.sh to set paths, then:
bash train.sh
```

Trained checkpoints will be saved to `checkpoint/`.

### Step 2 — Launch ChatEvac

```bash
python Agent.py
```

The GUI window will open. The workflow proceeds as follows:

1. **Upload** a building floorplan image via the "Select Image" button
2. The LLM triggers **feature extraction** — the Diffusion model segments the floorplan into walls, indoor areas, and exits
3. Review the extracted feature map (clickable thumbnail)
4. Confirm or modify **simulation parameters** (occupant count, speed, dimensions, etc.) in natural language
5. The LLM triggers the **SFM simulation** — real-time pedestrian evacuation visualization
6. Launch the **data analysis dashboard** to view evacuation curves and density heatmaps
7. The LLM generates a **regulatory-grounded evaluation report** with design optimization recommendations


## Important Note

> [!NOTE]
> **Paper Under Review:** This repository accompanies the manuscript **EAAI-25-24082: "ChatEvac: End-to-End Automation Assessment for Building Evacuation Safety Based on Large Language and Diffusion Models,"** currently under review at *Engineering Applications of Artificial Intelligence (EAAI)*.
>
> The code in this repository represents the **architectural framework** of ChatEvac. The **trained ControlNet model weights** and the **complete 1,500-case annotated floorplan dataset** will be released upon formal publication of the paper.
>

## Citation

If you use ChatEvac in your research, please cite:

```bibtex
@article{lu2025chatevac,
  title={ChatEvac: End-to-End Automation Assessment for Building Evacuation Safety
         Based on Large Language and Diffusion Models},
  author={Lu, Tong and Ding, Saizhe and Zhang, Yuxin and Deng, Rong and Huang, Xinyan},
  journal={Engineering Applications of Artificial Intelligence},
  note={Under review},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

---

*For questions or collaboration inquiries, please contact the corresponding authors.*
