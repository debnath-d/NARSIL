import json
import logging
import os
from enum import StrEnum, auto
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from matplotlib import ticker
from matplotlib.collections import LineCollection
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

torch.set_printoptions(precision=3, sci_mode=False)


class TSPDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path = None,
        graph_size: int = None,
        dataset_size=0,
        heatmap_path: Path = None,
    ):
        super().__init__()

        if isinstance(dataset_path, Path):
            assert dataset_path.is_file(), f"{dataset_path} does not exist"
            self.data = torch.load(dataset_path, map_location="cpu")
            if dataset_size > 0:
                self.data = {key: val[:dataset_size] for key, val in self.data.items()}
            if "tours" in self.data and self.data["tours"].dtype != torch.int:
                self.data["tours"] = self.data["tours"].int()
        elif isinstance(graph_size, int) and dataset_size > 0:
            self.generate_tsp(dataset_size, graph_size)
        else:
            raise ValueError("dataset_size must be a positive integer")
        if isinstance(heatmap_path, Path):
            assert heatmap_path.is_file(), f"{heatmap_path} does not exist"
            self.data["heatmaps"] = torch.from_numpy(np.load(heatmap_path)["arr_0"])
            if dataset_size > 0:
                self.data["heatmaps"] = self.data["heatmaps"][:dataset_size]

            assert self.data["heatmaps"].shape[:2] == self.data["graphs"].shape[:2], (
                f"Invalid heatmaps shape: {self.data['heatmaps'].shape}"
            )

    def __len__(self):
        return len(self.data["graphs"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}

    def generate_tsp(self, dataset_size, graph_size):
        self.data = {"graphs": torch.rand(dataset_size, graph_size, 2)}


class TrainMode(StrEnum):
    SELF_IMPROVEMENT = auto()
    SUPERVISED = auto()


# Setup distributed environment
def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # world_size = int(os.environ.get("WORLD_SIZE", 1))

    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        # rank = dist.get_rank()
        # local_rank = dist.get_node_local_rank()
        # world_size = dist.get_world_size()
    return rank, local_rank


# Configure logging
def setup_logging(rank, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            # logging.StreamHandler(),
            logging.FileHandler(
                filename=output_dir / f"rank_{rank}.log",
                mode="w",
            ),
        ],
    )


# Get checkpoint if exists
def get_checkpoint(
    device: torch.device,
    path: str | Path,
):
    if isinstance(path, (str, Path)) and Path(path).is_file():
        checkpoint: dict[str] = torch.load(
            path, map_location=device, weights_only=False
        )

        config_path = None
        for parent in Path(path).parents[1::-1]:
            if (parent / "train.config").is_file():
                config_path = parent / "train.config"
                break
        checkpoint_config: dict[str, dict[str]] = (
            json.loads(config_path.read_text()) if isinstance(config_path, Path) else {}
        )
        return checkpoint, checkpoint_config
    return None, None


# Load checkpoint if exists
def load_checkpoint(
    checkpoint: dict[str],
    model: nn.Module,
    optimizer: optim.Optimizer = None,
    scheduler: optim.lr_scheduler.LRScheduler = None,
):
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"] + 1


def save_checkpoint(checkpoint: dict, path: Path):
    if path.is_file():
        tmp_path = path.with_suffix(".tmp")
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(path)  # Ensure atomic write
    else:
        torch.save(checkpoint, path)


def generate_random_rotation_matrices(n: int, device="cpu"):
    # Generate n random angles in radians
    theta = torch.rand(n, device=device) * 2 * torch.pi

    # Create rotation matrices of shape (n, 2, 2)
    cos_theta = theta.cos()
    sin_theta = theta.sin()

    x_basis = torch.stack([cos_theta, sin_theta], dim=-1)  # Shape: (n, 2)
    y_basis = torch.stack([-sin_theta, cos_theta], dim=-1)  # Shape: (n, 2)

    return torch.stack([x_basis, y_basis], dim=-1)  # Shape: (n, 2, 2)


def transform_graphs(graphs: torch.Tensor, K=1, R: torch.Tensor = None):
    """
    Create K versions of each graph in G (batch_size, graph_size, 2) by:
    - Normalizing coordinates to centroid at origin
    - Keeping the original graph
    - Creating K-1 copies, flipping half of them about y-axis
    - Applying random rotations to all K-1 copies

    Args:
        graphs: Input tensor of shape (batch_size, graph_size, 2)
        K: Total number of versions per graph (including original)
        R: Optional rotation matrices of shape ((K - 1) * batch_size, 2, 2).

    Returns:
        Tensor of shape (batch_size, K, graph_size, 2) containing K versions per graph
    """

    batch_size, graph_size, _ = graphs.shape

    # Normalize coordinates by subtracting the center
    graphs = graphs - graphs.mean(dim=-2, keepdim=True)

    graphs = graphs.unsqueeze(1)

    if K <= 1:
        return graphs  # If K is 1, return the original normalized graphs

    # Create K-1 copies of the normalized graphs
    transformed = graphs.repeat(1, K - 1, 1, 1).reshape(-1, graph_size, 2)

    n_transformed = transformed.shape[0]

    # Flip half of the graphs about the y-axis (Multiply x-coordinates by -1)
    transformed[torch.arange(n_transformed) % 2 == 0, :, 0] *= -1

    # Generate random rotation matrices for (K-1) * batch_size graphs
    if isinstance(R, torch.Tensor):
        R = R[:n_transformed]
        assert R.shape == (n_transformed, 2, 2), (
            f"R must have shape ({K - 1} * {batch_size}, 2, 2)"
        )
    else:
        R = generate_random_rotation_matrices(n_transformed, device=graphs.device)

    transformed = (transformed @ R.mT).reshape(batch_size, K - 1, graph_size, 2)

    # Combine original normalized graph with transformed versions
    return torch.cat([graphs, transformed], dim=1)


@torch.inference_mode()
def batched_two_opt(tour: torch.Tensor, cost_matrix: torch.Tensor, max_iterations=1000):
    batch_size, graph_size = tour.shape
    batch_idx = torch.arange(batch_size, device=tour.device)[:, None, None]
    iterations = torch.zeros(batch_size, dtype=torch.int, device=tour.device)
    not_done = torch.ones_like(iterations, dtype=torch.bool)
    for _ in range(max_iterations):
        _batch_idx = batch_idx[not_done]
        _tour = tour[not_done]
        A_ij = cost_matrix[_batch_idx, _tour.unsqueeze(-1), _tour.unsqueeze(-2)]
        A_i_plus_1_j_plus_1 = A_ij.roll(shifts=(-1, -1), dims=(-1, -2))

        A_i_i_plus_1 = cost_matrix[
            _batch_idx,
            _tour.unsqueeze(-1),
            _tour.roll(shifts=-1, dims=-1).unsqueeze(-1),
        ]
        A_j_j_plus_1 = A_i_i_plus_1.reshape(-1, 1, graph_size)

        change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
        valid_change = change.triu(diagonal=2).reshape(_batch_idx.shape[0], -1)
        batch_min_change = valid_change.min(dim=-1).values
        flatten_argmin_idx = valid_change.argmin(dim=-1)
        min_i = flatten_argmin_idx.floor_divide(graph_size)
        min_j = flatten_argmin_idx.remainder(graph_size)

        for batch, i, j, min_change in zip(
            _batch_idx.flatten(), min_i, min_j, batch_min_change
        ):
            if min_change > -1e-6:
                not_done[batch] = False
                continue
            tour[batch, i + 1 : j + 1] = tour[batch, i + 1 : j + 1].flip(-1)
            iterations[batch] += 1
        if not not_done.any():
            break

    final_tour_cost = cost_matrix[
        torch.arange(batch_size).unsqueeze(-1),
        tour,
        tour.roll(shifts=-1, dims=-1),
    ].sum(-1)
    return tour, final_tour_cost, iterations


def adj_list_to_matrix(adj_list: torch.Tensor):
    *batch_dimensions, graph_size, edge_list_size = adj_list.shape
    batch_size = int(np.prod(batch_dimensions))
    adj_list = adj_list.reshape(-1, graph_size, edge_list_size)

    # Initialize the adjacency matrix with zeros
    adj_matrix = torch.zeros(
        batch_size, graph_size, graph_size, dtype=torch.bool, device=adj_list.device
    ).scatter(dim=-1, index=adj_list, src=torch.ones_like(adj_list, dtype=torch.bool))

    # Reshape back to original batch dimensions
    return adj_matrix.reshape(*batch_dimensions, graph_size, graph_size)


def tour_to_adj_list(tours: torch.Tensor):
    *batch_dimensions, graph_size = tours.shape
    batch_size = int(np.prod(batch_dimensions))
    tours = tours.reshape(-1, graph_size)

    batch_idx = torch.arange(batch_size, device=tours.device)[:, None]
    adj_list = torch.zeros(
        batch_size, graph_size, 2, dtype=torch.int, device=tours.device
    )
    adj_list[batch_idx, tours, 0] = tours.roll(shifts=1, dims=-1)
    adj_list[batch_idx, tours, 1] = tours.roll(shifts=-1, dims=-1)

    # Reshape back to original batch dimensions
    return adj_list.reshape(*batch_dimensions, graph_size, 2)


def adj_list_to_tour(adj_list: torch.Tensor):
    """
    Warning! edge_list must be valid TSP
    """
    *batch_dimensions, graph_size, _ = adj_list.shape
    batch_size = int(np.prod(batch_dimensions))
    adj_list = adj_list.reshape(-1, graph_size, 2)
    batch_idx = torch.arange(batch_size, dtype=torch.int, device=adj_list.device)

    tour = [
        torch.zeros(batch_size, dtype=torch.int, device=adj_list.device),
        adj_list[:, 0, 0],
    ]
    for i in range(graph_size - 2):
        next_candidates = adj_list[batch_idx, tour[-1]]
        next_node = next_candidates != tour[-2].unsqueeze(-1)
        tour.append(next_candidates[next_node])
    return torch.stack(tour, dim=-1).reshape(*batch_dimensions, graph_size)


def edge_list_to_adj_list(edge_list: torch.Tensor):
    *batch_dimensions, graph_size, _ = edge_list.shape
    batch_size = int(np.prod(batch_dimensions))
    batch_idx = torch.arange(batch_size, device=edge_list.device)[:, None, None]
    edge_list = edge_list.reshape(batch_size, graph_size, 2)
    edge_list_flip = edge_list.flip(-1)

    adj_list = torch.zeros_like(edge_list)
    adj_list[batch_idx, edge_list, 0] = edge_list_flip
    z = (adj_list[batch_idx, edge_list, 0] != edge_list_flip).int()
    adj_list[batch_idx, edge_list, z] = edge_list_flip
    return adj_list.reshape(*batch_dimensions, graph_size, 2)


@torch.inference_mode()
def plot_graph(
    graph: torch.Tensor,
    probs: torch.Tensor = None,
    edges: torch.Tensor = None,
    baseline_edges: torch.Tensor = None,
    title="Graph Visualization",
    fig: plt.Figure = None,
    node_size=50,
    figsize=(20, 10),
):
    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        axes = fig.axes
    for ax in axes:
        ax.set_aspect("equal")

    if graph.min() < 0:
        lim_min = min(graph.min().item(), -0.7)
        lim_max = max(graph.max().item(), 0.7)
    else:
        lim_min = -0.05
        lim_max = 1.05

    graph = graph.cpu().numpy()

    N = len(graph)

    if isinstance(probs, torch.Tensor):
        if edges is None:
            # If no selected edges, use the top 2 higheset prob neighbors for each node
            edges = torch.argsort(probs, descending=True)[:, :2]

        # Create heatmap of probabilities
        rank_matrix = 1 + torch.argsort(probs, descending=True).argsort().cpu().numpy()
        probs = probs.cpu().numpy()

        im = axes[1].imshow(probs, vmin=0, vmax=1, cmap="coolwarm_r", aspect="auto")

        # Add text annotations for each cell
        if N < 20:
            color = np.where((probs > 0.2) & (probs < 0.8), "black", "white")
            for i in range(N):
                for j in range(N):
                    axes[1].text(
                        x=j,
                        y=i,
                        s=str(rank_matrix[i, j]),
                        ha="center",
                        va="center",
                        fontsize=10,
                        color=color[i, j],
                    )

        axes[1].set_title("Model Predicted Probabilities")
        axes[1].set_xlabel("Node Index")
        axes[1].set_ylabel("Node Index")
        axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        axes[1].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Add colorbar
        fig.colorbar(im, ax=axes[1], label="Probability")

    axes[0].scatter(*graph.T, s=node_size, alpha=0.7, c="blue", zorder=10)

    for i, node in enumerate(graph):
        axes[0].annotate(
            str(i),
            node,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8,
        )
    if isinstance(edges, torch.Tensor):
        tour_coordinates = graph[edges]
        line_collection = LineCollection(tour_coordinates, colors="black", zorder=0)
        axes[0].add_collection(line_collection)

    if isinstance(baseline_edges, torch.Tensor):
        edges = torch.stack(
            [torch.arange(N).to(baseline_edges.device).repeat(2, 1).mT, baseline_edges],
            dim=-1,
        ).reshape(-1, 2)
        undirected = edges[:, 0] < edges[:, 1]
        undirected_edges = graph[edges[undirected].cpu().numpy()]
        # Draw baseline edges
        for x, y in undirected_edges.mT:
            axes[0].plot(x, y, linestyle="--", color="black", alpha=0.4, linewidth=1)

    axes[0].set_title(title)
    axes[0].set_xlabel("X Coordinate")
    axes[0].set_ylabel("Y Coordinate")
    axes[0].set_xlim(lim_min, lim_max)
    axes[0].set_ylim(lim_min, lim_max)

    fig.tight_layout()
    fig.show()
    fig.savefig("graph_visualization.png")
    # plt.close(fig)


@torch.inference_mode()
def probs_two_selections(probs: torch.Tensor):
    p_by_one_minus_p = probs / (1 - probs).clamp_min(1e-10)
    S = p_by_one_minus_p.sum(-1, keepdim=True)
    probs_two_draws = probs * (1 - p_by_one_minus_p + S)
    return probs_two_draws


@torch.inference_mode()
def select_seq_edge(probs: torch.Tensor, cost_matrix: torch.Tensor = None):
    k = probs.shape[0] // cost_matrix.shape[0]
    batch_size, graph_size, _ = probs.shape
    device = probs.device

    triu_mask = torch.ones(
        graph_size, graph_size, dtype=torch.bool, device=device
    ).tril()

    clamp_threshold = 1e-6  # IMPORTANT! Increase further if needed
    probs = probs * (probs > 0.001)
    if not torch.equal(probs, probs.mT):
        probs = probs * probs.mT
    probs = probs.half().clamp_min(clamp_threshold)
    probs.masked_fill_(mask=triu_mask, value=0)

    if isinstance(cost_matrix, torch.Tensor):
        cost_matrix = cost_matrix.half()
        cost_matrix.masked_fill_(mask=triu_mask, value=torch.inf)
        cost_matrix = (
            cost_matrix.unsqueeze(1)
            .repeat(1, k, 1, 1)
            .reshape(-1, graph_size, graph_size)
        )

    if isinstance(cost_matrix, torch.Tensor):
        assert probs.shape == cost_matrix.shape

    batch_idx = torch.arange(batch_size, dtype=torch.int, device=device).unsqueeze(-1)
    other_end = torch.arange(graph_size, dtype=torch.int, device=device).repeat(
        batch_size, 1
    )
    node_degree = torch.zeros_like(other_end, dtype=torch.uint8)

    tour_edges = torch.zeros(
        (batch_size, graph_size, 2), dtype=torch.int, device=device
    )
    for step in range(graph_size):
        sel_probs, idx = probs.view(batch_size, -1).max(-1)
        idx = idx.int()
        i_idx = idx.floor_divide(graph_size)
        j_idx = idx.remainder(graph_size)

        if isinstance(cost_matrix, torch.Tensor):
            zero_probs = sel_probs <= clamp_threshold

            if zero_probs.any():
                idx = cost_matrix.view(batch_size, -1).min(-1).indices[zero_probs].int()
                i_idx[zero_probs] = idx.floor_divide(graph_size)
                j_idx[zero_probs] = idx.remainder(graph_size)

        tour_edges[:, step, 0] = i_idx
        tour_edges[:, step, 1] = j_idx
        current_edge = tour_edges[:, step]

        node_degree[batch_idx, current_edge] += 1

        assert node_degree.max() <= 2, "Node degree exceeded 2"

        current_node_is_degree_2 = node_degree[batch_idx, current_edge].flatten() == 2
        degree_2_node = current_edge.flatten()[current_node_is_degree_2]
        degree_2_batch_idx = batch_idx.expand(-1, 2).flatten()[current_node_is_degree_2]

        probs[degree_2_batch_idx, degree_2_node] = 0
        probs[degree_2_batch_idx, :, degree_2_node] = 0
        probs[batch_idx.squeeze(), *current_edge.T] = 0
        if isinstance(cost_matrix, torch.Tensor):
            cost_matrix[degree_2_batch_idx, degree_2_node] = torch.inf
            cost_matrix[degree_2_batch_idx, :, degree_2_node] = torch.inf
            cost_matrix[batch_idx.squeeze(), *current_edge.T] = torch.inf

        other_end_current_edge = other_end[batch_idx, current_edge].sort().values

        if node_degree.sum(-1).eq(2 * (graph_size - 1)).all():
            tour_edges[:, step + 1] = other_end_current_edge
            break

        other_end[batch_idx, other_end_current_edge] = other_end_current_edge.flip(-1)

        probs[batch_idx.squeeze(), *other_end_current_edge.T] = 0
        if isinstance(cost_matrix, torch.Tensor):
            cost_matrix[batch_idx.squeeze(), *other_end_current_edge.T] = torch.inf

    return tour_edges


def self_improvement_learning(
    config: dict,
    probs: torch.Tensor,
    cost_matrix: torch.Tensor,
):
    batch_size, *_, graph_size = probs.shape
    n_unique_graphs = cost_matrix.shape[0]

    tour_edges = select_seq_edge(probs=probs, cost_matrix=cost_matrix)

    tour_edges: torch.Tensor
    *_, n_seq, graph_size, _ = tour_edges.shape

    tour_adj_list = edge_list_to_adj_list(tour_edges)
    tours = adj_list_to_tour(tour_adj_list)
    two_opt_tours, two_opt_cost, _ = batched_two_opt(
        tour=tours[:, 0],
        cost_matrix=cost_matrix.unsqueeze(1)
        .expand(-1, batch_size // n_unique_graphs, -1, -1)
        .reshape(batch_size, graph_size, graph_size),
        max_iterations=1000,
    )
    two_opt_cost = two_opt_cost.reshape(n_unique_graphs, -1)
    two_opt_tours = two_opt_tours.reshape(n_unique_graphs, -1, graph_size)

    tour_edges = tour_edges.reshape(n_unique_graphs, -1, n_seq, graph_size, 2)
    tour_costs = cost_matrix[
        torch.arange(n_unique_graphs)[:, None, None, None],
        *tour_edges.movedim(-1, 0),
    ].sum(-1)
    greedy_cost = tour_costs[..., 0]

    batch_idx = torch.arange(n_unique_graphs)

    best_tour_idx = two_opt_cost.argmin(-1)
    best_cost = two_opt_cost[batch_idx, best_tour_idx]
    best_tour_adj_list = tour_to_adj_list(two_opt_tours[batch_idx, best_tour_idx])
    greedy_best_gap = 100 * (greedy_cost / best_cost.unsqueeze(-1) - 1)

    probs = probs.reshape(n_unique_graphs, -1, graph_size, graph_size)

    best_tour_probs = probs.gather(
        dim=-1,
        index=best_tour_adj_list.int()
        .unsqueeze(1)
        .expand(-1, config["k_transforms"], -1, -1),
    ).clamp_min(1e-10)

    loss = best_tour_probs.log().mean().neg()

    metrics = {
        "Target Cost": best_cost,
        "Target Greedy Gap": greedy_best_gap,
    }

    return loss, metrics


def supervised(
    probs: torch.Tensor,
    batch_data: dict[str, torch.Tensor],
):
    target_tours = batch_data["tours"].to(probs.device)
    target_tour_cost = batch_data["tour_costs"].to(probs.device)
    target_tour_adj_list = (
        tour_to_adj_list(target_tours).unsqueeze(1).expand(-1, probs.shape[1], -1, -1)
    )

    loss = (
        probs.gather(dim=-1, index=target_tour_adj_list)
        .clamp_min(1e-10)
        .log()
        .sum(-1)
        .mean()
        .neg()
        / 2
    )
    metrics = {
        "Target Tour Cost": target_tour_cost.mean().item(),
    }

    return loss, metrics
