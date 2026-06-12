"""Script to generate the Colab notebook programmatically via nbformat."""

import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells = [
    # --- Title ---
    new_markdown_cell(
        "# BIU DRL Ex03 — Fitness RL: LSTM + REINFORCE + A2C\n"
        "**Course:** Deep Reinforcement Learning — Dr. Yoram Segal, Bar-Ilan University\n\n"
        "This notebook implements all 6 parts of the assignment end-to-end."
    ),

    # --- Setup ---
    new_markdown_cell("## Setup"),
    new_code_cell(
        "# Install the package (when running in Colab)\n"
        "# !pip install -q git+https://github.com/hodayakashh/biu-fitness-training-ex03.git\n\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n\n"
        "from fitness_rl import FitnessRLSDK\n\n"
        "sdk = FitnessRLSDK('config/setup.json')\n"
        "print('SDK initialised.')"
    ),

    # --- Part B: Dataset ---
    new_markdown_cell(
        "## Part B — Dataset\n\n"
        "Search Kaggle for a structured workout log dataset "
        "(e.g. *gym workout tracker*, *gym members exercise tracking*).\n\n"
        "**Download with Kaggle API:**\n"
        "```bash\n"
        "!pip install -q kaggle\n"
        "# Upload kaggle.json first, then:\n"
        "!kaggle datasets download -d <owner>/<dataset>\n"
        "!unzip <file>.zip -d data/\n"
        "```\n\n"
        "Update `config/setup.json → data.columns` to match your CSV column names."
    ),
    new_code_cell(
        "CSV_PATH = 'data/workout.csv'  # update this path\n\n"
        "# Verify the file exists\n"
        "from pathlib import Path\n"
        "assert Path(CSV_PATH).exists(), f'File not found: {CSV_PATH}'"
    ),

    # --- Part C: Data Pipeline ---
    new_markdown_cell(
        "## Part C — Data Pipeline & LSTM Transition Model\n\n"
        "The pipeline converts raw exercise records into daily summaries, "
        "clusters days into 6 workout-type actions, normalises state features, "
        "and builds sliding-window tensors for LSTM training."
    ),
    new_code_cell(
        "data = sdk.prepare_data(CSV_PATH)\n\n"
        "print(f'Training windows : {data[\"X_train\"].shape}')\n"
        "print(f'Validation windows: {data[\"X_val\"].shape}')\n"
        "print(f'Action clusters  : {data[\"kmeans\"].n_clusters}')"
    ),
    new_code_cell(
        "lstm_result = sdk.train_lstm(data)\n\n"
        "print(f'Epochs trained   : {len(lstm_result[\"train_losses\"])}')\n"
        "print(f'Final train MSE  : {lstm_result[\"train_losses\"][-1]:.4f}')\n"
        "print(f'Final val MSE    : {lstm_result[\"val_losses\"][-1]:.4f}')"
    ),

    # --- Part D: REINFORCE ---
    new_markdown_cell(
        "## Part D — REINFORCE\n\n"
        "Train a policy π_θ(a|s) using episodic Monte Carlo returns G_t "
        "with a mean baseline to reduce gradient variance."
    ),
    new_code_cell(
        "reinforce_result = sdk.train_reinforce(lstm_result['model'], data)\n\n"
        "returns = reinforce_result['episode_returns']\n"
        "print(f'Episodes trained : {len(returns)}')\n"
        "print(f'First 10 avg return: {sum(returns[:10])/10:.3f}')\n"
        "print(f'Last  10 avg return: {sum(returns[-10:])/10:.3f}')"
    ),

    # --- Part E: A2C ---
    new_markdown_cell(
        "## Part E — A2C (Advantage Actor-Critic)\n\n"
        "Train Actor and Critic networks with per-step TD updates. "
        "The Critic V_ψ(s) provides a low-variance baseline: "
        "δ_t = r_t + γV(s') − V(s)."
    ),
    new_code_cell(
        "a2c_result = sdk.train_a2c(lstm_result['model'], data)\n\n"
        "returns_a2c = a2c_result['episode_returns']\n"
        "print(f'Episodes trained : {len(returns_a2c)}')\n"
        "print(f'First 10 avg return: {sum(returns_a2c[:10])/10:.3f}')\n"
        "print(f'Last  10 avg return: {sum(returns_a2c[-10:])/10:.3f}')"
    ),

    # --- Part F: Plots ---
    new_markdown_cell(
        "## Part F — Analysis & Plots\n\n"
        "All 5 required figures are generated and saved to `results/plots/`."
    ),
    new_code_cell(
        "paths = sdk.save_all_plots(lstm_result, reinforce_result, a2c_result, data)\n"
        "print('Saved figures:')\n"
        "for p in paths:\n"
        "    print(f'  {p}')"
    ),
    new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "import matplotlib.image as mpimg\n\n"
        "for p in paths:\n"
        "    img = mpimg.imread(str(p))\n"
        "    fig, ax = plt.subplots(figsize=(9, 4))\n"
        "    ax.imshow(img); ax.axis('off'); ax.set_title(p.name)\n"
        "    plt.tight_layout(); plt.show()"
    ),

    # --- Analysis ---
    new_markdown_cell(
        "## Analysis\n\n"
        "### 1. Dataset and representation\n"
        "_Was the dataset appropriate? What required feature engineering?_\n\n"
        "### 2. LSTM model quality\n"
        "_Did the LSTM produce plausible next-state predictions?_\n\n"
        "### 3. REINFORCE versus A2C\n"
        "_Convergence speed, stability, final return comparison._\n\n"
        "### 4. Practical interpretation\n"
        "_What training strategy did the policy learn?_\n\n"
        "### 5. Limitations and improvements\n"
        "_What assumptions are simplistic? How could the design be improved?_"
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "version": "3.10.0"}

out = Path(__file__).parent / "ex03_fitness_rl.ipynb"
nbformat.write(nb, out)
print(f"Notebook written to {out}")
