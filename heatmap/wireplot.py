import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_wire_dotplot(circuit_str, num_wires, x_label_str, save_path):
    gates = [g.strip() for g in circuit_str.split(";") if g.strip()]
    
    total_counts = np.zeros(num_wires, dtype=int)
    active_counts = np.zeros(num_wires, dtype=int)

    for gate in gates:
        wires = [int(w) for w in gate]
        for i, w in enumerate(wires):
            total_counts[w] += 1
            if i == 0: 
                active_counts[w] += 1

    x = np.arange(num_wires)

    plt.figure(figsize=(14, 6))
    plt.scatter(x, total_counts, color="blue", s=20, label="Total gates")
    plt.scatter(x, active_counts, color="red", s=20, label="Active gates")

    plt.xlabel(f"Wire Index ({x_label_str})")
    plt.ylabel("Gate Count")
    plt.title("Gate Counts per Wire")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved wire dot plot to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate wire dot plot from circuit file")
    parser.add_argument("--c", type=str, required=True, help="Path to circuit file")
    parser.add_argument("--n", type=int, required=True, help="Number of wires")
    parser.add_argument("--x", type=str, required=True, help="Label for X-axis")
    args = parser.parse_args()

    with open(args.c, "r") as f:
        circuit_str = f.read().strip()

    output = "wire_dotplot.png"

    print("Generating wire dot plot...")
    plot_wire_dotplot(circuit_str, args.n, args.x, output)
