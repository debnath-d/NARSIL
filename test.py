import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import NARSILModel
from utilities import (
    TSPDataset,
    adj_list_to_tour,
    batched_two_opt,
    edge_list_to_adj_list,
    get_checkpoint,
    load_checkpoint,
    select_seq_edge,
    transform_graphs,
)

torch.set_printoptions(precision=3, sci_mode=False)


@torch.inference_mode()
def test_batch(
    config: dict,
    batch_data: dict[str, torch.Tensor],
    device: torch.device,
):
    graphs, heatmaps, target_tour, target_cost = (
        batch_data[x].to(device) for x in ("graphs", "heatmaps", "tours", "tour_costs")
    )
    cost_matrix = torch.cdist(graphs, graphs, p=2)

    batch_size, graph_size, _ = graphs.shape

    solution_times = {}

    start = time.perf_counter()  # greedy solution
    tour_edges = select_seq_edge(probs=heatmaps, cost_matrix=cost_matrix)
    end = time.perf_counter()
    solution_times["Greedy Time"] = end - start

    *_, n_seq, graph_size, _ = tour_edges.shape

    tour_cost = cost_matrix[
        torch.arange(batch_size)[:, None, None],
        *tour_edges.movedim(-1, 0),
    ].sum(-1)

    tour_cost = tour_cost.reshape(batch_size, -1)
    tour_edges = tour_edges.reshape(batch_size, -1, graph_size, 2)

    tour_adj_list = edge_list_to_adj_list(tour_edges)
    tour_greedy_adj_list = tour_adj_list[:, 0]

    tours = adj_list_to_tour(tour_greedy_adj_list)

    start = time.perf_counter()
    _, two_opt_greedy_cost, iterations = batched_two_opt(
        tour=tours,
        cost_matrix=cost_matrix,
        max_iterations=1000,
    )
    end = time.perf_counter()
    solution_times["2-Opt Time"] = end - start

    cost = torch.stack([tour_cost[:, 0], two_opt_greedy_cost], dim=-1)

    if cost.ndim == 2:
        target_cost = target_cost.unsqueeze(1)
    metrics = {
        "Cost": cost.sum(dim=0),
        "Optimality Gap": (cost / target_cost - 1).sum(dim=0),
        "2-Opt Iterations": iterations.sum(),
    }

    return metrics | solution_times


@torch.inference_mode()
def test_epoch(config: dict, dataloader: DataLoader, device: torch.device):
    agg = {}
    for batch_data in dataloader:
        metrics = test_batch(
            config=config,
            batch_data=batch_data,
            device=device,
        )
        for metric, value in metrics.items():
            if metric not in agg:
                agg[metric] = value
            else:
                agg[metric] += value
    batch_size, graph_size, _ = batch_data["graphs"].shape
    dataset_size = len(dataloader.dataset)

    metrics = {
        metric: value
        * (1 if metric in {"Cost", "2-Opt Iterations"} else 100)
        / dataset_size
        for metric, value in agg.items()
        if "Time" not in metric
    }
    for metric in ("Cost", "Optimality Gap"):
        metrics[f"{metric} Greedy"], metrics[f"{metric} 2-Opt"] = metrics[metric]
        del metrics[metric], agg[metric]

    metrics = {metric: value.item() for metric, value in metrics.items()}

    return agg | metrics


@torch.inference_mode()
def evaluate_heatmaps(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heatmaps_dir = Path(config["general"]["heatmaps_dir"])
    dataset_dir = Path(config["general"]["dataset_dir"])
    output_dir = Path(config["general"]["outputs_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config['general']['method_name']}.md"
    if output_path.is_file():
        print(f"Skip existing file: {output_path}")
        return
    testing_params = config["testing"]
    pbar = tqdm(list(heatmaps_dir.glob("tsp*.npz")))
    assert len(pbar) > 0, f"No heatmaps found in {heatmaps_dir}"
    metrics = {}
    for heatmap_path in pbar:
        pbar.set_description(f"Evaluating {heatmap_path.name}")
        try:
            graph_size, diffusion_steps, *_ = heatmap_path.stem.split("_")
            diffusion_steps = int(diffusion_steps.replace("steps", ""))
        except Exception:
            graph_size, *_ = heatmap_path.stem.split("_")
        graph_size = int(graph_size.replace("tsp", ""))
        dataset = TSPDataset(
            dataset_path=dataset_dir / f"tsp{graph_size}_test_concorde.pt",
            heatmap_path=heatmap_path,
        )
        batch_size = int(testing_params["nodes_per_batch"] // graph_size)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        metrics[f"TSP{graph_size}"] = test_epoch(
            config=testing_params,
            dataloader=dataloader,
            device=device,
        )
    results_df = (
        pd.DataFrame(metrics)
        .sort_index()
        .sort_index(axis=1, key=lambda cols: cols.str.removeprefix("TSP").astype(int))
    )
    heatmaps_time_file = heatmaps_dir / "time_stats.csv"
    if heatmaps_time_file.is_file():
        heatmaps_time = pd.read_csv(heatmaps_time_file).set_index("Unnamed: 0").T
        results_df = pd.concat([results_df, heatmaps_time])
        results_df.loc["Heatmap+Greedy Time"] = results_df.loc[
            ["Heatmap Generation Time", "Greedy Time"]
        ].sum()
        results_df.loc["Total Time"] = results_df.loc[
            ["Heatmap Generation Time", "Greedy Time", "2-Opt Time"]
        ].sum()
    results_df.round(3).to_markdown(output_path)
    results_df.filter(like="Time", axis=0).round(3).to_markdown(
        output_dir / "time_stats.md"
    )
    results_df.to_csv(output_path.with_suffix(".csv"))


@torch.inference_mode()
def generate_heatmaps(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heatmaps_dir = Path(config["general"]["heatmaps_dir"])
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(config["general"]["dataset_dir"])
    assert dataset_dir.is_dir(), f"{dataset_dir} not found"

    checkpoint, checkpoint_config = get_checkpoint(
        path=config["general"].get("load_checkpoint", None),
        device=device,
    )
    assert checkpoint is not None, "Checkpoint not found!"

    # Merge checkpoint config with provided config
    if isinstance(checkpoint_config, dict):
        config_groups = set(config.keys()).union(set(checkpoint_config.keys()))
        config = {
            group: checkpoint_config.get(group, {}) | config.get(group, {})
            for group in config_groups
        }

    model = NARSILModel(**config["model"]).to(device)
    load_checkpoint(checkpoint=checkpoint, model=model)
    model.eval()

    heatmap_generation_time = {}

    pbar = tqdm(list(dataset_dir.glob("tsp*concorde.pt")))
    assert len(pbar) > 0, f"No datasets found in {dataset_dir}"

    for dataset_path in pbar:
        pbar.set_description(f"Processing {dataset_path.name}")
        graph_size = int(dataset_path.stem.split("_")[0].replace("tsp", ""))
        if graph_size not in {1000}:
            continue
        save_path = heatmaps_dir / f"tsp{graph_size}.npz"
        if save_path.is_file():
            print(f"Skip existing file for TSP{graph_size}")
            continue
        dataset = TSPDataset(dataset_path=dataset_path)
        # batch_size = int(config["testing"]["nodes_per_batch"] // graph_size)
        batch_size = len(dataset)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        heatmaps = []
        start = time.perf_counter()
        for batch_data in dataloader:
            graphs = batch_data["graphs"].to(device)
            graphs = transform_graphs(graphs).squeeze()
            batch_heatmaps = model(graphs)
            heatmaps.append(batch_heatmaps.cpu())
        end = time.perf_counter()
        heatmap_generation_time[f"TSP{graph_size}"] = end - start
        heatmaps = torch.cat(heatmaps, dim=0) if len(heatmaps) > 1 else heatmaps[0]
        heatmaps[heatmaps < 0.001] = 0  # For better compression
        np.savez_compressed(save_path, heatmaps.numpy())
    pd.Series(
        data=heatmap_generation_time,
        name="Heatmap Generation Time",
    ).to_markdown(heatmaps_dir / "time_stats.md")


def merge_evaluations(config: dict):
    results_dir = Path(config["general"]["outputs_dir"])
    output_dir = results_dir / "merged"
    if output_dir.is_dir():
        print(f"Skipping. Outputs directory already exists: {output_dir}")
        return
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    result_files = list(results_dir.glob("*.csv"))
    dfs = {}
    for result_file in result_files:
        df = pd.read_csv(result_file, index_col=0)
        dfs[result_file.stem] = df
    df_concat = pd.concat(dfs.values(), keys=dfs.keys())
    stacked = df_concat.stack()
    reshaped = stacked.unstack(level=2).sort_index(
        axis=1, key=lambda cols: cols.str.removeprefix("TSP").astype(int)
    )

    for metric, df in reshaped.groupby(level=1):
        output_file = (
            output_dir / f"{metric.lower().replace(' ', '_').replace('/', '_')}.md"
        )
        df = df.droplevel(level=1)
        df.round(3).to_markdown(output_file)
        df.to_csv(output_file.with_suffix(".csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config JSON file",
    )
    args = parser.parse_args()

    # Load configuration
    config_path: Path = args.config
    default_config_path = Path("configs/test.json")
    if isinstance(config_path, Path):
        if config_path.is_file():
            print(f"Loading config from {config_path}")
        else:
            print(f"Config file at {config_path} does not exist. Exiting.")
            return
    else:
        print(
            f"No config file provided. Trying to load config from {default_config_path}"
        )
        if default_config_path.is_file():
            config_path = default_config_path
        else:
            default_config_path.parent.mkdir(parents=True, exist_ok=True)
            default_config_path.touch()
            print(
                f"Created default config file at {default_config_path}. Please add your configuration there."
            )
            return

    config = json.loads(config_path.read_text())
    # generate_heatmaps(config)
    # generate_softdist_heatmaps(config)
    evaluate_heatmaps(config)
    # merge_evaluations(config)
    # test(config)


if __name__ == "__main__":
    main()
