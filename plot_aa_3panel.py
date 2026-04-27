"""
3-panel figure for AA-mode rewiring runs:
  left    : initial network (directed graph, per-firm colors)
  right   : final network (same positions & colors, different arrows)
  center  : top    -- per-firm price trajectories vs round
            bottom -- scatter of (round, rewiring firm id), one dot per swap event

Runs two separate simulations -- b=1.0 (CRS) and b=0.9 -- and writes one
figure per run.
"""
import os
import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np

from test_convergence import generate_base_network, run_unified_simulation
from utils import generate_a_parameter, generate_parameter


# ---- run config --------------------------------------------------------------
N = 10
C = 1           # base connectivity (override: this sweep uses sparse networks)
CC = 1          # alternate suppliers per firm
AISI_SPREAD = 0.1
MAX_SWAPS = 1
MODE = "aa"
SEED = 42
NB_ROUNDS = 50

B_CASES = [
    (1.0, "b = 1.0 (CRS)", "figure_aa_b1.0.png"),
    (0.9, "b = 0.9",       "figure_aa_b0.9.png"),
]

NODE_COLORS = [
    "#D66B6B", "#6BD66B", "#6B6BD6", "#D69B6B", "#A66BA6",
    "#D6D66B", "#6BD6D6", "#D6A3A3", "#A66B6B", "#7F7F7F",
]


# ---- helpers -----------------------------------------------------------------
def edges_array_to_digraph(edges, n):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for src, dst in np.asarray(edges):
        G.add_edge(int(src), int(dst))
    return G


def draw_network(ax, G, positions, title):
    nx.draw_networkx_edges(
        G, pos=positions, ax=ax,
        edge_color="black", width=1, alpha=0.7,
        connectionstyle="arc3,rad=0.12",
        arrows=True, arrowsize=12, node_size=500,
    )
    nx.draw_networkx_nodes(
        G, pos=positions, ax=ax,
        node_color=[NODE_COLORS[i] for i in G.nodes],
        node_size=500, edgecolors="black", linewidths=1.0,
    )
    nx.draw_networkx_labels(G, pos=positions, ax=ax, font_size=12)
    ax.set_axis_off()
    ax.set_title(title, fontsize=13)


def run_one(b_value):
    random.seed(SEED)
    np.random.seed(SEED)
    b = generate_parameter({"mode": "homogeneous", "value": b_value}, N, "b", verbose=False)
    a = generate_a_parameter({"mode": "homogeneous", "value": 0.5}, b, N, verbose=False)
    z = generate_parameter({"mode": "homogeneous", "value": 1.0}, N, "z", verbose=False)
    base_state = generate_base_network(N, C, CC, AISI_SPREAD, seed=SEED, a=a, b=b)
    result = run_unified_simulation(
        base_state, a, b, z,
        mode=MODE, seed=SEED, max_swaps=MAX_SWAPS,
        nb_rounds=NB_ROUNDS, trace=True,
    )
    return result


def make_figure(result, title_suffix, out_path):
    trace = result["trace"]
    edges_initial = trace["edges"][0]
    edges_final = trace["edges"][-1]
    prices = np.array(trace["prices"])  # (K, N) -- K = 1 + #price-change steps + optional final anchor
    price_steps = np.array(trace["price_steps"])
    events = trace["rewire_events"]
    t_max = int(price_steps[-1]) if len(price_steps) else 0

    G0 = edges_array_to_digraph(edges_initial, N)
    Gf = edges_array_to_digraph(edges_final, N)

    # Shared positions: layout on union of initial and final edges
    G_union = nx.Graph()
    G_union.add_nodes_from(range(N))
    G_union.add_edges_from(G0.edges())
    G_union.add_edges_from(Gf.edges())
    positions = nx.kamada_kawai_layout(G_union)

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        width_ratios=[1.0, 1.35, 1.0],
        height_ratios=[2, 1],
        hspace=0.08, wspace=0.12,
    )
    ax_left = fig.add_subplot(gs[:, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 1], sharex=ax_top)
    ax_right = fig.add_subplot(gs[:, 2])

    draw_network(ax_left, G0, positions, "Initial network")
    draw_network(ax_right, Gf, positions, "Final network")

    for i in range(N):
        ax_top.step(price_steps, prices[:, i], where="post",
                    color=NODE_COLORS[i], lw=1.6, label=f"p{i}")
    ax_top.set_ylabel("Price $p_i$")
    ax_top.set_title(title_suffix, fontsize=13)
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(alpha=0.3)

    if events:
        xs = [e["t"] for e in events]
        ys = [e["firm"] for e in events]
        cols = [NODE_COLORS[e["firm"]] for e in events]
        ax_bot.scatter(xs, ys, c=cols, marker="s", s=55,
                       edgecolors="black", linewidths=0.6)
    ax_bot.set_xlabel("Time step $t$")
    ax_bot.set_ylabel("Rewiring firm id")
    ax_bot.set_ylim(-0.5, N - 0.5)
    ax_bot.set_yticks(range(N))
    #ax_bot.set_xlim(0, max(t_max, 1))
    ax_bot.set_xlim(0, 100)
    ax_bot.grid(alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"  converged={result['converged']}, rounds={result['rounds']}, "
          f"initial_utility={result['initial_utility']:.4f}, "
          f"final_utility={result['final_utility']:.4f}, "
          f"rewire_events={len(events)}")


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    for b_value, title, fname in B_CASES:
        result = run_one(b_value)
        make_figure(result, f"AA, max_swaps={MAX_SWAPS}, {title}",
                    os.path.join(out_dir, fname))


if __name__ == "__main__":
    main()
