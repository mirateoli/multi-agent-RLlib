import os
import matplotlib.pyplot as plt
import pandas as pd
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
import matplotlib.cm as cm
import numpy as np


# Specify the directory where the Ray Tune results are saved
results_dir = "C:\\Users\\MDO-Disco\\ray_results\\PPO_2024-06-28_11-22-55\\"

# Load the results using Ray Tune Analysis
analysis = ExperimentAnalysis(results_dir)

# Get all trial dataframes
dfs = analysis.trial_dataframes

# Function to plot mean reward for all trials
def plot_all_trials(dfs, title):
    plt.figure(figsize=(10, 6))

    # Create a list to store tuples of (trial_id, lr) for sorting
    trials_lr = [(trial_id, analysis.get_all_configs()[trial_id]["lr"]) for trial_id in dfs.keys()]

    # Sort trials by lr (ascending)
    trials_lr_sorted = sorted(trials_lr, key=lambda x: x[1])

    for trial_id, lr in trials_lr_sorted:
        df = dfs[trial_id]
        config = analysis.get_all_configs()[trial_id]
        sgd_minibatch_size = config["sgd_minibatch_size"]
        train_batch_size = config["train_batch_size"]

        # Format lr in scientific notation
        lr_label = f"{lr:.2e}"
        
        # Plot mean reward for each trial
        plt.plot(df["training_iteration"], df["episode_reward_mean"], label=f"lr={lr_label}, sgd_minibatch={sgd_minibatch_size}, train_batch={train_batch_size}")

    plt.xlabel("Training Iterations")
    plt.ylabel("Mean Episode Reward")
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    # Retrieve the best checkpoint for the best overall trial
    # Retrieve the best checkpoint and hyperparameters
    best_trial = analysis.get_best_trial(metric="episode_reward_mean", mode="max")
    print("Best trial: {}".format(best_trial)) 
    best_config = best_trial.config
    print("Best trial config: {}".format(best_config))
    best_checkpoint = best_trial.checkpoint
    print("Best checkpoint: {}".format(best_checkpoint))

    # Print the episode reward mean for the best checkpoint
    best_trial_df = dfs[best_trial.trial_id]
    best_checkpoint_reward = best_trial_df.loc[best_trial_df["episode_reward_mean"].idxmax()]["episode_reward_mean"]
    print("Episode reward mean for the best checkpoint: {}".format(best_checkpoint_reward))


# show specific hyperparameters

def plot_hyperparameter(df_dicts, hyperparameter, hyperparameter_values, title):
    plt.figure(figsize=(10, 6))
    
    colors = cm.rainbow(np.linspace(0, 1, len(hyperparameter_values)))
    value_color_map = {value: colors[i] for i, value in enumerate(hyperparameter_values)}

    best_trial_info = {value: {"trial_id": None, "max_mean_reward": -float('inf')} for value in hyperparameter_values}

    for value in hyperparameter_values:
        for trial_id, df in df_dicts.items():
            config_value = analysis.get_all_configs()[trial_id].get(hyperparameter)
            if config_value == value:
                max_mean_reward = df["episode_reward_mean"].max()  # Calculate rolling mean over last 10 iterations

                if max_mean_reward > best_trial_info[value]["max_mean_reward"]:
                    best_trial_info[value]["trial_id"] = trial_id
                    best_trial_info[value]["max_mean_reward"] = max_mean_reward

                if hyperparameter == "lr":
                    label = f"lr={analysis.get_all_configs()[trial_id]['lr']:.2e}"
                else:
                    label = f"{hyperparameter}={value}"
                plt.plot(df["training_iteration"], df["episode_reward_mean"], color=value_color_map[value], label=label)

    # To avoid multiple labels for the same hyperparameter value, we create a unique legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    plt.xlabel("Training Iterations")
    plt.ylabel("Mean Episode Reward")
    plt.title(title)
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    plt.grid(True)
    plt.show()

    # Print best performing trials
    print(f"Best performing trials based on highest mean reward:")
    for value, info in best_trial_info.items():
        if info["trial_id"] is not None:
            print(f"{hyperparameter}={value}: Trial {info['trial_id']} (Max Mean Reward: {info['max_mean_reward']:.2f})")

# Get unique values for each hyperparameter from the config
learning_rates = sorted(set(config["lr"] for config in analysis.get_all_configs().values()))
sgd_minibatch_sizes = sorted(set(config["sgd_minibatch_size"] for config in analysis.get_all_configs().values()))
training_batch_sizes = sorted(set(config["train_batch_size"] for config in analysis.get_all_configs().values()))

# Plot mean reward for all trials
plot_all_trials(dfs, "Mean Episode Reward vs Training Iterations for All Trials")

# Plot mean reward for different learning rates
plot_hyperparameter(dfs, "lr", learning_rates, "Mean Episode Reward vs Training Iterations for Different Learning Rates")

# Plot mean reward for different SGD minibatch sizes
plot_hyperparameter(dfs, "sgd_minibatch_size", sgd_minibatch_sizes, "Mean Episode Reward vs Training Iterations for Different SGD Minibatch Sizes")

# Plot mean reward for different training batch sizes
plot_hyperparameter(dfs, "train_batch_size", training_batch_sizes, "Mean Episode Reward vs Training Iterations for Different Training Batch Sizes")
