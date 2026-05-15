"""
Combined AA-mode rewiring figure: b_i = 1.0 and b_i = 0.9 stacked top-down
in a single output PNG (results/figures/figure_aa.png).

Layout:
    Top  (b_i = 1.0, CRS):  (a) Initial network | (b) Prices | (c) Rewire
                            events | (d) Final network
    Bot  (b_i = 0.9):       (e) Initial network | (f) Prices | (g) Rewire
                            events | (h) Final network

For the b_i = 0.9 case, each firm's alternate-supplier edges are overlaid
as dashed red arrows on both network panels so the reader sees the
candidate set against which the active suppliers (solid black) compete.

The dynamics use mode="aa", $\\kappa = 1$ swap per firm-evaluation.
"""
import os
import random
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np

# Allow `python scripts/plot_aa_3panel.py` from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from rewiring.networks import generate_base_network
from rewiring.parameters import generate_a_parameter, generate_parameter
from rewiring.simulation import run_unified_simulation


# ---- run config --------------------------------------------------------------
N = 10
C = 1           # base connectivity
CC = 1          # alternate suppliers per firm
AISI_SPREAD = 0.1
KAPPA = 1
MODE = "aa"
SEED = 42
NB_ROUNDS = 50

B_CASES = [
    (1.0, r"$b_i = 1.0$ (CRS)"),
    (0.9, r"$b_i = 0.9$"),
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


def pool_edges_from_base_state(base_state, n):
    """Initial active+alternate edges of the pool (invariant under swaps).

    Returns a list of (supplier, buyer) tuples.
    """
    sup = base_state["supplier_id_list"]
    alt = base_state["alternate_supplier_id_list"]
    pool = []
    for buyer in range(n):
        for s in sup[buyer]:
            pool.append((int(s), int(buyer)))
        for s in alt[buyer]:
            pool.append((int(s), int(buyer)))
    return pool


def alt_digraph(active_edges, pool, n):
    """Edges in pool but not currently active."""
    active_set = {(int(s), int(d)) for s, d in np.asarray(active_edges)}
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for s, d in pool:
        if (s, d) not in active_set:
            G.add_edge(s, d)
    return G


def draw_network(ax, G, positions, alt_G=None, title=None):
    if alt_G is not None and alt_G.number_of_edges() > 0:
        nx.draw_networkx_edges(
            alt_G, pos=positions, ax=ax,
            edge_color="red", style="dashed", width=1.0, alpha=0.8,
            connectionstyle="arc3,rad=0.12",
            arrows=True, arrowsize=10, node_size=500,
        )
    nx.draw_networkx_edges(
        G, pos=positions, ax=ax,
        edge_color="black", width=1, alpha=0.85,
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
    if title:
        ax.set_title(title, fontsize=12, loc="left")


def run_one(b_value):
    random.seed(SEED)
    np.random.seed(SEED)
    b = generate_parameter({"mode": "homogeneous", "value": b_value},
                           N, "b", verbose=False)
    a = generate_a_parameter({"mode": "homogeneous", "value": 0.5},
                             b, N, verbose=False)
    z = generate_parameter({"mode": "homogeneous", "value": 1.0},
                           N, "z", verbose=False)
    base_state = generate_base_network(N, C, CC, AISI_SPREAD,
                                       seed=SEED, a=a, b=b)
    result = run_unified_simulation(
        base_state, a, b, z,
        mode=MODE, seed=SEED, max_swaps=KAPPA,
        nb_rounds=NB_ROUNDS, trace=True,
    )
    return result, base_state


def make_combined_figure(cases, out_path):
    """cases = list of (b_value, b_label, result, base_state) tuples.

    Panel letters run sequentially through the cases top-to-bottom:
        (a) (b) (c) (d) for first case,
        (e) (f) (g) (h) for second.
    """
    fig = plt.figure(figsize=(14, 10))
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.22)
    panel_letters = iter("abcdefghijkl")

    for case_idx, (b_value, b_label, result, base_state) in enumerate(cases):
        trace = result["trace"]
        edges_initial = trace["edges"][0]
        edges_final   = trace["edges"][-1]
        prices = np.array(trace["prices"])
        price_steps = np.array(trace["price_steps"])
        events = trace["rewire_events"]

        G0 = edges_array_to_digraph(edges_initial, N)
        Gf = edges_array_to_digraph(edges_final, N)
        pool_edges = pool_edges_from_base_state(base_state, N)
        alt_G0 = alt_digraph(edges_initial, pool_edges, N)
        alt_Gf = alt_digraph(edges_final,   pool_edges, N)

        G_union = nx.Graph()
        G_union.add_nodes_from(range(N))
        G_union.add_edges_from(G0.edges())
        G_union.add_edges_from(Gf.edges())
        G_union.add_edges_from(pool_edges)
        positions = nx.kamada_kawai_layout(G_union)

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=outer[case_idx],
            width_ratios=[1.0, 1.35, 1.0],
            height_ratios=[2, 1],
            hspace=0.08, wspace=0.12,
        )
        ax_left  = fig.add_subplot(inner[:, 0])
        ax_top   = fig.add_subplot(inner[0, 1])
        ax_bot   = fig.add_subplot(inner[1, 1], sharex=ax_top)
        ax_right = fig.add_subplot(inner[:, 2])

        show_alt = (b_value == 0.9)

        # (.) Initial network
        letter_a = next(panel_letters)
        draw_network(ax_left, G0, positions,
                     alt_G=alt_G0 if show_alt else None,
                     title=f"({letter_a}) Initial network, {b_label}")

        # (.) Prices
        letter_b = next(panel_letters)
        for i in range(N):
            ax_top.step(price_steps, prices[:, i], where="post",
                        color=NODE_COLORS[i], lw=1.6, label=f"p{i}")
        ax_top.set_ylabel("Price $p_i$")
        ax_top.set_title(f"({letter_b}) Prices, {b_label}",
                         fontsize=12, loc="left")
        ax_top.tick_params(labelbottom=False)
        ax_top.grid(alpha=0.3)

        # (.) Rewire events
        letter_c = next(panel_letters)
        if events:
            xs = [e["t"] for e in events]
            ys = [e["firm"] for e in events]
            cols = [NODE_COLORS[e["firm"]] for e in events]
            ax_bot.scatter(xs, ys, c=cols, marker="s", s=55,
                           edgecolors="black", linewidths=0.6)
        ax_bot.set_xlabel(r"Time step $t$")
        ax_bot.set_ylabel("Rewiring firm id")
        ax_bot.set_ylim(-0.5, N - 0.5)
        ax_bot.set_yticks(range(N))
        ax_bot.set_xlim(0, 100)
        ax_bot.set_title(f"({letter_c}) Rewire events, {b_label}",
                         fontsize=12, loc="left")
        ax_bot.grid(alpha=0.3)

        # (.) Final network
        letter_d = next(panel_letters)
        draw_network(ax_right, Gf, positions,
                     alt_G=alt_Gf if show_alt else None,
                     title=f"({letter_d}) Final network, {b_label}")

        # Per-case stdout line
        print(f"  case {b_label}: converged={result['converged']}, "
              f"rounds={result['rounds']}, "
              f"initial_utility={result['initial_utility']:.4f}, "
              f"final_utility={result['final_utility']:.4f}, "
              f"rewire_events={len(events)}")

    fig.suptitle(rf"AA simulation, $\kappa = {KAPPA}$", fontsize=14, y=0.995)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              os.pardir))
    out_dir = os.path.join(repo_root, "results", "figures")
    os.makedirs(out_dir, exist_ok=True)

    cases = []
    for b_value, label in B_CASES:
        result, base_state = run_one(b_value)
        cases.append((b_value, label, result, base_state))

    out_path = os.path.join(out_dir, "figure_aa.png")
    make_combined_figure(cases, out_path)


if __name__ == "__main__":
    main()
