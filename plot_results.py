import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Sequence
import tyro


def main(
    level_paths: Sequence[str] = (
        "worlds_l_grasp_easy",
        "worlds_l_catapult",
        "worlds_l_cartpole_thrust",
        "worlds_l_hard_lunar_lander",
        "worlds_l_mjc_half_cheetah",
        "worlds_l_mjc_swimmer",
        "worlds_l_mjc_walker",
        "worlds_l_h17_unicycle",
        "worlds_l_chain_lander",
        "worlds_l_catcher_v3",
        "worlds_l_trampoline",
        "worlds_l_car_launch",
    ),
):
    EVAL_PATH = './logs-eval-n04/'
    FILE_NAME = '/results.csv'
    OUTPUT_DIR = './logs-figs/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for level_path in level_paths:
        TRIAL_NAME = level_path
        FILE_PATH = EVAL_PATH + TRIAL_NAME + FILE_NAME

        if not os.path.exists(FILE_PATH):
            print(f"Warning: {FILE_PATH} does not exist, skipping.")
            continue

        df = pd.read_csv(FILE_PATH)

        # Keep only rows with the minimum execute_horizon for each (delay, method) pair
        filtered = (
            df.loc[
                df.groupby(['delay', 'method'])['execute_horizon'].idxmin()
            ]
        )

        plt.figure(figsize=(8, 6))
        sns.lineplot(
            data=df,
            x='delay',
            y='returned_episode_solved',
            hue='method',
            marker='o'
        )

        plt.title(f'Delay vs. Returned Episode Solved for {TRIAL_NAME}')
        plt.xlabel('Delay')
        plt.ylabel('Returned Episode Solved')
        plt.ylim(0.0, 1.05)
        plt.xticks(ticks=sorted(df['delay'].unique()))
        plt.legend(title='Method')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{TRIAL_NAME}.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    tyro.cli(main)
