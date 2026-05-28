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

# Allow `python campaigns/aa_illustrative_ts/plot.py` from anywhere. This script
# lives 2 levels below the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir)))

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


def run_one(b_value, detect_cycles=True):
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
        detect_cycles=detect_cycles,
    )
    return result, base_state


def _label_inside(ax, text, where='top-left', fontsize=14):
    """Stamp '(letter)' inside the axes, in the requested corner."""
    if where == 'top-left':
        x, y, ha = 0.02, 0.97, 'left'
    elif where == 'top-right':
        x, y, ha = 0.98, 0.97, 'right'
    elif where == 'top-center':
        x, y, ha = 0.5, 0.97, 'center'
    else:
        raise ValueError(where)
    ax.text(x, y, text, transform=ax.transAxes,
            ha=ha, va='top', fontsize=fontsize, fontweight='bold')


def make_combined_figure(cases, out_path):
    """cases = list of (b_value, b_label, result, base_state) tuples.

    Panel letters run sequentially through the cases top-to-bottom:
        (a) (b) (c) (d) for the top case,
        (e) (f) (g) (h) for the bottom case.
    Network panels carry their letter top-left; time-series panels top-right.

    The bottom-case final network only highlights the *cycling* edges
    (symmetric difference between the last two visited supplier
    configurations) as dashed red -- exposing the period-k limit cycle
    rather than the full alternate-supplier set.
    """
    fig = plt.figure(figsize=(14, 10))
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.22)
    panel_letters = iter("abcdefghijkl")
    # Price-panel titles per case (top vs bottom).
    price_titles = {1.0: r"CRS ($b_i = 1$)", 0.9: r"DRS ($b_i = 0.9$)"}
    T_MAX_PLOT = 100  # x-axis extent for time-series panels in every case

    for case_idx, (b_value, b_label, result, base_state) in enumerate(cases):
        is_last_case = (case_idx == len(cases) - 1)

        trace = result["trace"]
        edges_initial = trace["edges"][0]
        edges_final   = trace["edges"][-1]
        prices = np.array(trace["prices"])
        price_steps = np.array(trace["price_steps"])
        events = trace["rewire_events"]

        # Extend the price trajectory to t = T_MAX_PLOT so step-plots reach
        # the right edge of the panel even when the run halted earlier
        # (limit cycle / fixed point).
        if len(price_steps) > 0 and price_steps[-1] < T_MAX_PLOT:
            price_steps = np.append(price_steps, T_MAX_PLOT)
            prices = np.vstack([prices, prices[-1]])

        # Cycling edges = symmetric difference of the last two visited
        # supplier configurations. Non-empty iff the last round had any
        # accepted swap -- which catches both detected period-k cycles and
        # ongoing-cycle runs in which detection was disabled.
        cycle_period = result.get('cycle_period')
        cycling_edges = set()
        if len(trace["edges"]) >= 2:
            final = {(int(s), int(d)) for s, d in trace["edges"][-1]}
            prev  = {(int(s), int(d)) for s, d in trace["edges"][-2]}
            diff = final ^ prev
            if diff:
                cycling_edges = diff

        G0 = edges_array_to_digraph(edges_initial, N)
        Gf_full = edges_array_to_digraph(edges_final, N)
        pool_edges = pool_edges_from_base_state(base_state, N)

        G_union = nx.Graph()
        G_union.add_nodes_from(range(N))
        G_union.add_edges_from(G0.edges())
        G_union.add_edges_from(Gf_full.edges())
        G_union.add_edges_from(pool_edges)
        positions = nx.kamada_kawai_layout(G_union)

        # 4-column inner grid: col 1 is an invisible spacer giving the
        # left network panel extra breathing room from the time-series.
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 4, subplot_spec=outer[case_idx],
            width_ratios=[1.0, 0.15, 1.35, 1.0],
            height_ratios=[2, 1],
            hspace=0.08, wspace=0.08,
        )
        ax_left  = fig.add_subplot(inner[:, 0])
        ax_top   = fig.add_subplot(inner[0, 2])
        ax_bot   = fig.add_subplot(inner[1, 2], sharex=ax_top)
        ax_right = fig.add_subplot(inner[:, 3])

        # ---- (.) Initial network (NO red alt edges in either case) ----
        draw_network(ax_left, G0, positions, alt_G=None)
        _label_inside(ax_left, f"({next(panel_letters)}) initial",
                      'top-left')

        # ---- (.) Prices (CRS / DRS title, no x-label) ----
        for i in range(N):
            ax_top.step(price_steps, prices[:, i], where="post",
                        color=NODE_COLORS[i], lw=1.6, label=f"p{i}")
        ax_top.set_ylabel("Price $p_i$")
        ax_top.set_title(price_titles.get(b_value, b_label),
                         fontsize=13, loc="center")
        ax_top.tick_params(labelbottom=False)
        ax_top.set_xlim(0, T_MAX_PLOT)
        ax_top.grid(alpha=0.3)
        _label_inside(ax_top, f"({next(panel_letters)})", 'top-center')

        # ---- (.) Rewire events (x-label only on the bottom case) ----
        if events:
            xs = [e["t"] for e in events]
            ys = [e["firm"] for e in events]
            cols = [NODE_COLORS[e["firm"]] for e in events]
            ax_bot.scatter(xs, ys, c=cols, marker="s", s=55,
                           edgecolors="black", linewidths=0.6)
        if is_last_case:
            ax_bot.set_xlabel(r"Time step $t$")
        ax_bot.set_ylabel("Rewiring firm id")
        ax_bot.set_ylim(-0.5, N - 0.5)
        ax_bot.set_yticks(range(N))
        ax_bot.set_xlim(0, T_MAX_PLOT)
        ax_bot.grid(alpha=0.3)
        _label_inside(ax_bot, f"({next(panel_letters)})", 'top-center')

        # ---- (.) Final network ----
        # If the run ended in a period-k cycle, highlight only the cycling
        # edges (the symmetric difference between the last two snapshots).
        # Stable active edges are drawn solid black; cycling edges -- one is
        # currently active, the other currently alternate -- are both
        # dashed red regardless of their current active/alternate status.
        if cycling_edges:
            final_set = {(int(s), int(d)) for s, d in edges_final}
            stable_active = final_set - cycling_edges
            Gf_stable = nx.DiGraph()
            Gf_stable.add_nodes_from(range(N))
            for s, d in stable_active:
                Gf_stable.add_edge(s, d)
            alt_Gf_cycle = nx.DiGraph()
            alt_Gf_cycle.add_nodes_from(range(N))
            for s, d in cycling_edges:
                alt_Gf_cycle.add_edge(s, d)
            draw_network(ax_right, Gf_stable, positions, alt_G=alt_Gf_cycle)
        else:
            draw_network(ax_right, Gf_full, positions, alt_G=None)
        _label_inside(ax_right, f"({next(panel_letters)}) final",
                      'top-left')

        # Per-case stdout line
        print(f"  case {b_label}: converged={result['converged']}, "
              f"cycle_period={cycle_period}, rounds={result['rounds']}, "
              f"initial_utility={result['initial_utility']:.4f}, "
              f"final_utility={result['final_utility']:.4f}, "
              f"rewire_events={len(events)}, "
              f"cycling_edges={len(cycling_edges)}")

    # No suptitle (per user spec).
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    # Tracked code in campaigns/aa_illustrative_ts/; output figures land in the
    # mirroring results/aa_illustrative_ts/ (gitignored).
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              os.pardir, os.pardir))
    out_dir = os.path.join(repo_root, "results", "aa_illustrative_ts")
    os.makedirs(out_dir, exist_ok=True)

    cases = []
    for b_value, label in B_CASES:
        # For the b=0.9 case we want the rewire-events panel to keep
        # logging real swaps past the first detected period-2 cycle, so we
        # disable cycle detection. The CRS case still uses detection (it
        # converges to a fixed point in a handful of rounds).
        detect_cycles = (b_value != 0.9)
        result, base_state = run_one(b_value, detect_cycles=detect_cycles)
        cases.append((b_value, label, result, base_state))

    out_path = os.path.join(out_dir, "figure_aa.png")
    make_combined_figure(cases, out_path)


if __name__ == "__main__":
    main()
