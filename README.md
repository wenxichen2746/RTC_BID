Simulated experiments for the paper [Real-Time Execution of Action Chunking Flow Policies](https://pi.website/download/real_time_chunking.pdf).

## Installation

```bash
# Clone Kinetix submodule
git submodule update --init
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Install dependencies
uv sync
```

## Reproduce results
Note that, for all scripts, your number of GPUs must divide the number of levels (default 12) because computation is
sharded over levels.

1. Train expert policies: `uv run src/train_expert.py`
    - By default, this will train 8 seeds per level for 65 million environment steps each.
    - Checkpoints, videos, and stats are written to a wandb project called `rtc-kinetix-expert` and the local directory `./logs-expert/<wandb-run-name>`. It is recommended to control other wandb options, like the run name, using environment variables.
2. Generate data: `uv run src/generate_data.py --config.run-path ./logs-expert/<wandb-run-name>`
    - For each level, this will automatically load the best-performing checkpoint for each seed (discarding seeds that didn't reach a certain success threshold). 
    - By default, 1 million environment steps are collected for each level using a mixture of expert policies.
    - Data is written back to `./logs-expert/<wandb-run-name>/data/`.
3. Train imitation learning policies: `uv run src/train_flow.py --config.run-path ./logs-expert/<wandb-run-name>`
    - This will load the data from step 2 and train flow matching policies for each level.
    - Checkpoints, videos, and stats are written to a wandb project called `rtc-kinetix-bc` and the local directory `./logs-bc/<wandb-run-name>`. It is recommended to control other wandb options, like the run name, using environment variables.
4. Evaluate imitation learning policies: `uv run src/eval_flow.py --config.run-path ./logs-bc/<wandb-run-name> --output-dir <output-dir>`
    - This will load the checkpoints from step 3 and evaluate them for 2048 trials per level by default.
    - Currently, the script performs an exhaustive sweep over inference delay and execution horizon for all methods.
