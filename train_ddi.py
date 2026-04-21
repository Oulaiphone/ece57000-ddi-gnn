"""
ECE 57000 - Artificial Intelligence | Course Project
Predicting Drug-Drug Interactions Using Graph Neural Networks

Author: Oulaiphone
Purdue University, Spring 2026

This script implements and compares GraphSAGE and GAT architectures
for DDI link prediction on the OGB ogbl-ddi benchmark.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import numpy as np
import time
import json
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Model Definitions
# ============================================================

class DDI_GraphSAGE(torch.nn.Module):
    """
    3-layer GraphSAGE encoder for DDI prediction.
    Uses mean aggregation over sampled neighborhoods.
    
    Written by: Oulaiphone (adapted from OGB examples)
    """
    def __init__(self, in_ch, hidden, out_ch, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, out_ch)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x


class DDI_GAT(torch.nn.Module):
    """
    3-layer Graph Attention Network encoder for DDI prediction.
    Uses multi-head attention to learn neighbor importance weights.
    
    Written by: Oulaiphone
    """
    def __init__(self, in_ch, hidden, out_ch, heads=4, dropout=0.3):
        super().__init__()
        # Multi-head attention: each head learns different interaction patterns
        self.conv1 = GATConv(in_ch, hidden // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout)
        # Final layer: single head for output
        self.conv3 = GATConv(hidden, out_ch, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x


# ============================================================
# Training and Evaluation Functions
# ============================================================

def train(model, x, edge_index, train_pos, optimizer, batch_size=65536):
    """
    Train one epoch with negative sampling.
    
    Positive edges are sampled from known DDIs; negative edges are
    randomly generated drug pairs assumed to be non-interacting.
    Uses BCE loss with dot-product scoring.
    
    Written by: Oulaiphone (adapted from OGB examples)
    """
    model.train()
    h = model(x, edge_index)

    # Sample positive edges
    perm = torch.randperm(train_pos.size(0))[:batch_size]
    pos_edge = train_pos[perm].t()

    # Sample negative edges (random drug pairs)
    neg_edge = torch.randint(0, x.size(0), pos_edge.size(), device=x.device)

    # Dot-product scoring
    pos_score = (h[pos_edge[0]] * h[pos_edge[1]]).sum(dim=-1)
    neg_score = (h[neg_edge[0]] * h[neg_edge[1]]).sum(dim=-1)

    # Binary cross-entropy loss
    loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
    loss -= torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, x, edge_index, pos_edge, neg_edge, evaluator):
    """
    Evaluate model using OGB Hits@20 metric.
    
    Written by: Oulaiphone (adapted from OGB examples)
    """
    model.eval()
    h = model(x, edge_index)

    pos_edge_t = pos_edge.t()
    neg_edge_t = neg_edge.t()

    pos_score = (h[pos_edge_t[0]] * h[pos_edge_t[1]]).sum(dim=-1)
    neg_score = (h[neg_edge_t[0]] * h[neg_edge_t[1]]).sum(dim=-1)

    results = evaluator.eval({
        'y_pred_pos': pos_score,
        'y_pred_neg': neg_score,
    })

    return results


# ============================================================
# Experiment Runner
# ============================================================

def run_experiment(name, model, x_emb, edge_index, train_pos,
                   val_pos, val_neg, test_pos, test_neg,
                   evaluator, lr=0.005, epochs=500, batch_size=65536,
                   eval_every=50, device='cpu'):
    """
    Complete train-evaluate cycle for fair model comparison.
    
    Re-initializes embeddings with Xavier uniform before each run
    to ensure both models start from the same random baseline.
    
    Written by: Oulaiphone
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {name}")
    print(f"{'='*60}")

    # Fresh embeddings for fair comparison
    torch.nn.init.xavier_uniform_(x_emb.weight)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(x_emb.parameters()),
        lr=lr
    )

    losses = []
    val_hits_history = []
    test_hits_history = []
    best_val = 0.0
    best_test = 0.0
    best_epoch = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        loss = train(model, x_emb.weight, edge_index, train_pos,
                     optimizer, batch_size)
        losses.append(loss)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:4d}/{epochs} | Loss: {loss:.4f}")

        if epoch % eval_every == 0:
            val_results = evaluate(model, x_emb.weight, edge_index,
                                   val_pos, val_neg, evaluator)
            test_results = evaluate(model, x_emb.weight, edge_index,
                                    test_pos, test_neg, evaluator)

            val_hits = val_results['hits@20']
            test_hits = test_results['hits@20']

            val_hits_history.append((epoch, val_hits))
            test_hits_history.append((epoch, test_hits))

            print(f"  ** Eval @ epoch {epoch}: Val Hits@20 = {val_hits:.4f}, "
                  f"Test Hits@20 = {test_hits:.4f}")

            if val_hits > best_val:
                best_val = val_hits
                best_test = test_hits
                best_epoch = epoch
                print(f"     -> New best! Val={best_val:.4f}, Test={best_test:.4f}")

    elapsed = time.time() - start_time

    results = {
        'name': name,
        'best_val_hits20': best_val,
        'best_test_hits20': best_test,
        'best_epoch': best_epoch,
        'final_loss': losses[-1],
        'training_time_sec': elapsed,
        'training_time_hr': elapsed / 3600,
        'losses': losses,
        'val_hits_history': val_hits_history,
        'test_hits_history': test_hits_history,
    }

    print(f"\n  Results for {name}:")
    print(f"    Best Val Hits@20:  {best_val:.4f} (epoch {best_epoch})")
    print(f"    Best Test Hits@20: {best_test:.4f}")
    print(f"    Training time:     {elapsed/3600:.2f} hours")

    return results


# ============================================================
# Plotting Utilities
# ============================================================

def plot_training_curves(all_results, output_dir='results'):
    """
    Plot training loss curves and Hits@20 progression.
    
    Written by: Oulaiphone
    """
    os.makedirs(output_dir, exist_ok=True)

    # Loss curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for res in all_results:
        ax.plot(res['losses'], label=res['name'], alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Curves — GraphSAGE vs GAT on ogbl-ddi')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/loss_curves.png")

    # Hits@20 progression
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for res in all_results:
        epochs_list = [e for e, _ in res['val_hits_history']]
        hits_list = [h for _, h in res['val_hits_history']]
        ax.plot(epochs_list, hits_list, 'o-', label=f"{res['name']} (Val)")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Hits@20')
    ax.set_title('Validation Hits@20 Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'hits20_progression.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/hits20_progression.png")


def print_comparison_table(all_results):
    """Print a formatted comparison table."""
    print(f"\n{'='*70}")
    print(f"{'Method':<25} {'Val Hits@20':>12} {'Test Hits@20':>13} {'Time (hr)':>10}")
    print(f"{'-'*70}")
    for res in all_results:
        print(f"{res['name']:<25} {res['best_val_hits20']:>11.2%} "
              f"{res['best_test_hits20']:>12.2%} {res['training_time_hr']:>10.2f}")
    print(f"{'='*70}")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='DDI Prediction with GNNs on OGB ogbl-ddi')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden/embedding dimension (default: 512)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate (default: 0.005)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs (default: 500)')
    parser.add_argument('--batch_size', type=int, default=65536,
                        help='Training batch size (default: 65536)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    parser.add_argument('--eval_every', type=int, default=50,
                        help='Evaluate every N epochs (default: 50)')
    parser.add_argument('--models', type=str, default='both',
                        choices=['graphsage', 'gat', 'both'],
                        help='Which model(s) to run (default: both)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory for output files (default: results)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ---- Load dataset ----
    print("\nLoading OGB ogbl-ddi dataset...")
    dataset = PygLinkPropPredDataset(name='ogbl-ddi')
    data = dataset[0]
    split_edge = dataset.get_edge_split()

    train_pos = split_edge['train']['edge']
    val_pos = split_edge['valid']['edge']
    val_neg = split_edge['valid']['edge_neg']
    test_pos = split_edge['test']['edge']
    test_neg = split_edge['test']['edge_neg']

    edge_index = data.edge_index.to(device)
    train_pos = train_pos.to(device)
    val_pos = val_pos.to(device)
    val_neg = val_neg.to(device)
    test_pos = test_pos.to(device)
    test_neg = test_neg.to(device)

    num_nodes = data.num_nodes
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Train edges: {train_pos.size(0)}")
    print(f"  Val edges:   {val_pos.size(0)}")
    print(f"  Test edges:  {test_pos.size(0)}")

    evaluator = Evaluator(name='ogbl-ddi')
    all_results = []

    # ---- Run GraphSAGE ----
    if args.models in ['graphsage', 'both']:
        x_emb = torch.nn.Embedding(num_nodes, args.hidden_dim).to(device)
        sage_model = DDI_GraphSAGE(
            args.hidden_dim, args.hidden_dim, args.hidden_dim,
            dropout=args.dropout
        ).to(device)

        sage_results = run_experiment(
            name='GraphSAGE (tuned)',
            model=sage_model,
            x_emb=x_emb,
            edge_index=edge_index,
            train_pos=train_pos,
            val_pos=val_pos,
            val_neg=val_neg,
            test_pos=test_pos,
            test_neg=test_neg,
            evaluator=evaluator,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            device=device,
        )
        all_results.append(sage_results)

    # ---- Run GAT ----
    if args.models in ['gat', 'both']:
        x_emb = torch.nn.Embedding(num_nodes, args.hidden_dim).to(device)
        gat_model = DDI_GAT(
            args.hidden_dim, args.hidden_dim, args.hidden_dim,
            heads=4, dropout=args.dropout
        ).to(device)

        gat_results = run_experiment(
            name='GAT (4 heads)',
            model=gat_model,
            x_emb=x_emb,
            edge_index=edge_index,
            train_pos=train_pos,
            val_pos=val_pos,
            val_neg=val_neg,
            test_pos=test_pos,
            test_neg=test_neg,
            evaluator=evaluator,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            device=device,
        )
        all_results.append(gat_results)

    # ---- Output results ----
    print_comparison_table(all_results)
    plot_training_curves(all_results, args.output_dir)

    # Save results to JSON (without non-serializable data)
    os.makedirs(args.output_dir, exist_ok=True)
    json_results = []
    for res in all_results:
        json_results.append({
            'name': res['name'],
            'best_val_hits20': float(res['best_val_hits20']),
            'best_test_hits20': float(res['best_test_hits20']),
            'best_epoch': res['best_epoch'],
            'final_loss': res['final_loss'],
            'training_time_hr': res['training_time_hr'],
        })
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Saved: {args.output_dir}/results.json")

    print("\nDone!")


if __name__ == '__main__':
    main()
