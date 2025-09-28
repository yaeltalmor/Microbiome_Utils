import itertools

import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from temporal_transformer import TemporalTransformerClassifier, EventEmbAggregateMethod, \
    EventTimeEmbAggregateMethod, TimeEmbeddingType

sys.path.append('/Users/yaeltalmor/PycharmProjects/labUtils')
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)
import pickle
from dataclasses import dataclass

#----------------------------------RUN ONE MODEL---------------------------------------------


# --- Helper functions ---
def create_dataloaders(train_data, val_data, batch_size=32):
    train_dataset = TensorDataset(*train_data)
    val_dataset = TensorDataset(*val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def compute_pos_weight(labels):
    N_pos = labels.sum().item()
    N_neg = len(labels) - N_pos
    return torch.tensor([N_neg / N_pos], dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion, device, tb_writer=None, global_step=0):
    model.train()
    epoch_loss = 0.0

    for batch in loader:
        batch = [b.to(device) for b in batch]
        (event_idx_b, value_idx_b, numeric_value_b, value_type_mask_b,
         t_values_b, src_mask_b, metadata_idx_b, labels_b) = batch

        optimizer.zero_grad()
        logits = model(event_idx_b, value_idx_b, numeric_value_b,
                       value_type_mask_b, t_values_b, src_mask_b,
                       metadata_idx_b, inference=False)

        loss = criterion(logits.squeeze(-1), labels_b.float())
        loss.backward()
        optimizer.step()

        if tb_writer:
            tb_writer.add_scalar("Loss/train", loss.item(), global_step)
        global_step += 1
        epoch_loss += loss.item()

    return epoch_loss / len(loader), global_step


def evaluate_train_epoch(model, loader, device, criterion):
    model.eval()
    probs_train, y_train = [], []
    train_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = [b.to(device) for b in batch]
            (event_idx_b, value_idx_b, numeric_value_b, value_type_mask_b,
             t_values_b, src_mask_b, metadata_idx_b, labels_b) = batch

            logits = model(event_idx_b, value_idx_b, numeric_value_b,
                           value_type_mask_b, t_values_b, src_mask_b,
                           metadata_idx_b, inference=False)
            loss = criterion(logits.squeeze(-1), labels_b.float())
            train_loss += loss.item()

            probs = torch.sigmoid(logits)
            probs_train.append(probs.squeeze(-1).cpu().numpy())
            y_train.append(labels_b.cpu().numpy())

    probs_train = np.concatenate(probs_train)
    y_train = np.concatenate(y_train)
    auc = roc_auc_score(y_train, probs_train)
    ap = average_precision_score(y_train, probs_train)
    return train_loss / len(loader), auc, ap


def validate(model, loader, device, criterion):
    model.eval()
    probs_val, y_val = [], []
    val_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = [b.to(device) for b in batch]
            (event_idx_b, value_idx_b, numeric_value_b, value_type_mask_b,
             t_values_b, src_mask_b, metadata_idx_b, labels_b) = batch

            # --- 1. raw logits for loss ---
            logits = model(event_idx_b, value_idx_b, numeric_value_b,
                           value_type_mask_b, t_values_b, src_mask_b,
                           metadata_idx_b,
                           inference=False)  # inference=False for return_probs=False to have logits instead
            loss = criterion(logits.squeeze(-1), labels_b.float())
            val_loss += loss.item()

            # --- 2. probs for metrics ---
            probs = torch.sigmoid(logits)  # same as return_probs=True
            probs_val.append(probs.squeeze(-1).cpu().numpy())
            y_val.append(labels_b.cpu().numpy())

    probs_val = np.concatenate(probs_val)
    y_val = np.concatenate(y_val)
    auc = roc_auc_score(y_val, probs_val)
    ap = average_precision_score(y_val, probs_val)
    fpr, tpr, _ = roc_curve(y_val, probs_val)
    prec, rec, _ = precision_recall_curve(y_val, probs_val)

    return val_loss / len(loader), auc, ap, fpr, tpr, prec, rec


def save_checkpoint(model, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, "best_model.pt")
    torch.save(model.state_dict(), model_path)


# --- Main training function ---
def train_eval(model, train_data, val_data, max_epochs, lr, batch_size=32,
               patience=10, weight_decay=1e-5, log_dir="./runs/exp1",
               save_best_model=True):
    """
    Train + evaluate with early stopping, modular version.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, val_loader = create_dataloaders(train_data, val_data, batch_size)
    pos_weight = compute_pos_weight(train_data[-1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer reduces LR when validation stops improving, giving the model a chance to refine.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    # If you see the curves jump early and then plateau, try halving the LR (e.g. 1e-3 → 5e-4 or 1e-4 → 5e-5).
    # Also consider small warmup or cosine decay.
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    tb_writer = SummaryWriter(log_dir=log_dir) if save_best_model else None

    best_ap, best_auc, best_epoch = -np.inf, -np.inf, 0
    best_fpr, best_tpr, best_prec, best_rec = None, None, None, None
    patience_counter = 0
    global_step = 0

    for epoch in range(max_epochs):
        # Training
        train_loss, global_step = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                                  tb_writer, global_step, )
        if tb_writer:
            tb_writer.add_scalar("Loss/train_epoch", train_loss, epoch)

        # Train metrics (epoch-level)
        train_loss_eval, train_auc, train_ap = evaluate_train_epoch(model, train_loader, device, criterion)

        # Validation metrics (epoch-level)
        val_loss, val_auc, val_ap, fpr, tpr, prec, rec = validate(model, val_loader, device, criterion)

        # lower LR if no val_ap improvement
        scheduler.step(val_ap)

        # Log everything
        if tb_writer:
            tb_writer.add_scalar("Loss/val_epoch", val_loss, epoch)
            tb_writer.add_scalar("AUC/train", train_auc, epoch)
            tb_writer.add_scalar("AP/train", train_ap, epoch)
            tb_writer.add_scalar("AUC/val", val_auc, epoch)
            tb_writer.add_scalar("AP/val", val_ap, epoch)

        # Early stopping & checkpoint
        if val_ap > best_ap:
            best_ap, best_auc, best_epoch = val_ap, val_auc, epoch
            best_fpr, best_tpr, best_prec, best_rec = fpr, tpr, prec, rec
            patience_counter = 0
            if save_best_model:
                save_checkpoint(model, log_dir)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break  # TODO: return this for early stopping!!
            # continue

    if tb_writer:
        tb_writer.close()

    return best_auc, best_ap, best_epoch, best_fpr, best_tpr, best_prec, best_rec


#----------------------------------RUN MODEL SELECTION---------------------------------------------
def run_cv(model_params, idx, labels_np, config_name, save_best, run_tag,
           event_idx, value_idx, numeric_value, value_type_mask,
           t_values, src_mask, metadata_idx, labels,
           event_type_vocab, cat_value_vocab,
           max_epochs, patience, n_splits):

    all_aucs, all_aps, all_epochs = [], [], []
    all_fprs, all_tprs, all_precisions, all_recalls = [], [], [], []
    metadata_dim = model_params["metadata_size_and_data"][0]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for fold, (train_fold_idx, val_fold_idx) in enumerate(
        skf.split(idx.cpu().numpy(), labels_np[idx.cpu().numpy()])
    ):
        # Convert indices back to torch tensors
        tr_idx = idx[torch.as_tensor(train_fold_idx, device=idx.device)]
        va_idx = idx[torch.as_tensor(val_fold_idx, device=idx.device)]

        # Split tensors
        train_data = (
            event_idx[tr_idx], value_idx[tr_idx], numeric_value[tr_idx],
            value_type_mask[tr_idx], t_values[tr_idx], src_mask[tr_idx],
            metadata_idx[tr_idx] if metadata_dim else torch.zeros((len(tr_idx), 0)),
            labels[tr_idx]
        )
        val_data = (
            event_idx[va_idx], value_idx[va_idx], numeric_value[va_idx],
            value_type_mask[va_idx], t_values[va_idx], src_mask[va_idx],
            metadata_idx[va_idx] if metadata_dim else torch.zeros((len(va_idx), 0)),
            labels[va_idx]
        )

        # Init model fresh per fold
        model = TemporalTransformerClassifier(
            num_event_tokens=len(event_type_vocab),
            num_value_tokens=len(cat_value_vocab),
            event_d_model=model_params["initial_emb_dim"], temp_d_model=model_params["initial_emb_dim"],
            num_layers=model_params["capacity"]["num_layers"],
            num_heads=model_params["capacity"]["num_heads"],
            d_ff=model_params["capacity"]["d_ff"],
            dropout=model_params["capacity"]["dropout"],
            pooling=model_params["pooling"],
            cls_hidden=model_params["capacity"]["cls_hidden"],
            metadata_dim=metadata_dim,
            md_token=model_params["md_token"],
            event_emb_agg_method=model_params["event_emb_agg_method"],
            event_time_emb_agg_method=model_params["event_time_emb_agg_method"],
            time_emb_type=model_params["time_emb_type"]
        )

        auc, ap, best_epoch, fpr, tpr, prec, rec = train_eval(
            model, train_data, val_data,
            max_epochs=max_epochs, lr=model_params["capacity"]["lr"], batch_size=16,
            patience=patience, weight_decay=model_params["capacity"]["weight_decay"],
            log_dir=f"./runs/{config_name}{run_tag}_fold{fold}",
            save_best_model=save_best
        )

        all_aucs.append(auc); all_aps.append(ap); all_epochs.append(best_epoch)
        all_fprs.append(fpr); all_tprs.append(tpr)
        all_precisions.append(prec); all_recalls.append(rec)

    return all_aucs, all_aps, all_epochs, all_fprs, all_tprs, all_precisions, all_recalls


def grid_param_check(param_grid,
                     train_idx, labels_np,
                     event_idx, value_idx, numeric_value, value_type_mask,
                     t_values, src_mask, metadata_idx, labels,
                     event_type_vocab, cat_value_vocab,
                     n_bootstraps=50, n_splits=3, max_epochs=50, patience=10, results_dir="./results"):
    results_dict = {}

    for config_name, params in param_grid.items():
        all_aucs, all_aps, all_epochs = [], [], []
        all_fprs, all_tprs, all_precisions, all_recalls = [], [], [], []

        # (A) Original full run - save model version and dump results to file
        res = run_cv(params, train_idx, labels_np, config_name,
                     save_best=True, run_tag="",
                     event_idx=event_idx, value_idx=value_idx,
                     numeric_value=numeric_value, value_type_mask=value_type_mask,
                     t_values=t_values, src_mask=src_mask,
                     metadata_idx=params["metadata_size_and_data"][1], labels=labels,
                     event_type_vocab=event_type_vocab, cat_value_vocab=cat_value_vocab,
                     max_epochs=max_epochs, patience=patience, n_splits=n_splits)

        all_aucs += res[0];
        all_aps += res[1];
        all_epochs += res[2]
        all_fprs += res[3];
        all_tprs += res[4]
        all_precisions += res[5];
        all_recalls += res[6]

        # (B) Bootstraps
        for b in range(n_bootstraps):
            boot_sample = train_idx[torch.randint(
                low=0, high=len(train_idx), size=(len(train_idx),), device=train_idx.device
            )]

            res_b = run_cv(params, boot_sample, labels_np, config_name,
                           save_best=False, run_tag=f"_bootstrap{b}",
                           event_idx=event_idx, value_idx=value_idx,
                           numeric_value=numeric_value, value_type_mask=value_type_mask,
                           t_values=t_values, src_mask=src_mask,
                           metadata_idx=params["metadata_size_and_data"][1], labels=labels,
                           event_type_vocab=event_type_vocab, cat_value_vocab=cat_value_vocab,
                           max_epochs=max_epochs, patience=patience, n_splits=n_splits)

            all_aucs += res_b[0];
            all_aps += res_b[1];
            all_epochs += res_b[2]
            all_fprs += res_b[3];
            all_tprs += res_b[4]
            all_precisions += res_b[5];
            all_recalls += res_b[6]

        # Aggregate results
        pos_rate = labels_np[train_idx.cpu().numpy()].mean()  # fraction of positives

        mean_ap, std_ap = np.mean(all_aps), np.std(all_aps)
        ap_snr = (mean_ap - pos_rate) / (std_ap if std_ap > 0 else 1e-8)

        mean_auc, std_auc = np.mean(all_aucs), np.std(all_aucs)
        auc_snr = (mean_auc - 0.5) / (std_auc if std_auc > 0 else 1e-8)

        mean_epochs, std_epochs = np.mean(all_epochs), np.std(all_epochs)

        params["metadata_size_and_data"] = params["metadata_size_and_data"][2]
        results_dict[config_name] = {
            "params": f"{params}",
            "AP_SNR": ap_snr,
            "AP_mean": mean_ap,
            "AP_std": std_ap,
            "AUC_SNR": auc_snr,
            "AUC_mean": mean_auc,
            "AUC_std": std_auc,
            "Epoch_mean": mean_epochs,
            "Epoch_std": std_epochs,
            "AUCs": all_aucs, "APs": all_aps,
            "FPRs": all_fprs, "TPRs": all_tprs,
            "Precisions": all_precisions, "Recalls": all_recalls
        }

        # Save per config
        out_path = os.path.join(results_dir, f"{config_name}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(results_dict[config_name], f)

        # Remove after saving to avoid memory explosion
        results_dict[config_name].pop("AUCs", None)
        results_dict[config_name].pop("APs", None)
        results_dict[config_name].pop("FPRs", None)
        results_dict[config_name].pop("TPRs", None)
        results_dict[config_name].pop("Precisions", None)
        results_dict[config_name].pop("Recalls", None)

        print(f"{config_name}: "
              f"AP_SNR={ap_snr:.3f}, "
              f"AP={mean_ap:.3f} +- {std_ap:.3f}, "
              f"AUC={mean_auc:.3f} +- {std_auc:.3f}, "
              f"Epoch={mean_epochs:.1f} +- {std_epochs:.3f},")

    # Pick best by AP_SNR instead of AP_mean
    sorted_cfgs = sorted(results_dict.items(), key=lambda x: x[1]["AP_SNR"], reverse=True)

    print("\nTop 3 configs by AP_SNR:")
    for cfg, stats in sorted_cfgs[:3]:
        print(f"{cfg} → AP_SNR={stats['AP_SNR']:.3f}, "
              f"AP={stats['AP_mean']:.3f} ± {stats['AP_std']:.3f}")

    best_cfgs = sorted_cfgs[:3]  # return top 3

    # Save total results
    out_path = os.path.join(results_dir, f"results_dict.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results_dict, f)
    out_path = os.path.join(results_dir, f"best_cfgs.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(best_cfgs, f)

    return results_dict, best_cfgs


def evaluate_grid_results(results_dict):
    """
    results_dict: dict
        Keys = config names (from param_grid),
        Values = dicts with {"AUC_mean": mean_auc, "AUC_std": std_auc, "AP_mean": mean_ap, "AP_std": std_ap}
    plot: bool
        Whether to generate boxplots of the bootstrap results.
    """

    # Collect results for plotting
    auc_means, auc_stds, ap_means, ap_stds, configs = [], [], [], [], []
    for name, res in results_dict.items():
        auc_mean = res["AUC_mean"]
        auc_std = res["AUC_std"]
        ap_mean = res["AP_mean"]
        ap_std = res["AP_std"]

        auc_means.append(auc_mean)
        auc_stds.append(auc_std)
        ap_means.append(ap_mean)
        ap_stds.append(ap_std)
        configs.append(name)

    # If asked, plot boxplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    # AUC plot
    sns.barplot(
        x=auc_means, y=configs,
        xerr=auc_stds, capsize=0.3, ax=axes[0], color="skyblue"
    )
    axes[0].set_title("AUC (seeds & SCV mean ± std)", fontsize=14)
    axes[0].set_xlim(0.55, 0.85)  # tight range for your values
    axes[0].xaxis.set_major_locator(plt.MultipleLocator(0.01))  # grid every 0.01
    axes[0].grid(True, which="major", axis="x", linestyle="--", alpha=0.6)
    axes[0].tick_params(axis="x", rotation=45, labelsize=10)

    # AP plot
    sns.barplot(
        x=ap_means, y=configs,
        xerr=ap_stds, capsize=0.3, ax=axes[1], color="lightcoral"
    )
    axes[1].set_title("AP (seeds & SCV mean ± std)", fontsize=14)
    axes[1].set_xlim(0.2, 0.5)
    axes[1].xaxis.set_major_locator(plt.MultipleLocator(0.01))
    axes[1].grid(True, which="major", axis="x", linestyle="--", alpha=0.6)
    axes[1].tick_params(axis="x", rotation=45, labelsize=10)

    plt.tight_layout()
    plt.show()

    return

@dataclass
class TransformerBatch:
    t_values: torch.Tensor
    src_mask: torch.Tensor
    event_idx: torch.Tensor
    value_idx: torch.Tensor
    numeric_value: torch.Tensor
    value_type_mask: torch.Tensor
    metadata_weight_idx: torch.Tensor
    metadata_idx: torch.Tensor
    event_type_vocab: dict
    cat_value_vocab: dict
    labels: torch.Tensor

def load_data(NUM_YEARS: int, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_values = torch.load(
        f"/home/elhanan/PROJECTS/SHEBA_HABERMAN_YT/temporal_transformer/data/transformer__t_values__{NUM_YEARS}y.pt").to(device)
    src_mask = torch.load(
        f"/home/elhanan/PROJECTS/SHEBA_HABERMAN_YT/temporal_transformer/data/transformer__events_mask__{NUM_YEARS}y.pt").to(device)
    event_idx = torch.load(
        f"/home/elhanan/PROJECTS/SHEBA_HABERMAN_YT/temporal_transformer/data/transformer__event_idx__{NUM_YEARS}y.pt").to(device)
    value_idx = torch.load(
        f"/home/elhanan/PROJECTS/SHEBA_HABERMAN_YT/temporal_transformer/data/transformer__value_idx__{NUM_YEARS}y.pt").to(device)
    numeric_value = torch.load(
        f"/home/elhanan/PROJECTS/SHEBA_HABERMAN_YT/temporal_transformer/data/transformer__numeric_value__{NUM_YEARS}y.pt").to(device)
    value_type_mask = torch.load(
        f"/home/elhanan/PROJECTS/SHEBA_HABERMAN_YT/temporal_transformer/data/transformer__value_type_mask__{NUM_YEARS}y.pt").to(device)
    metadata_weight_idx = torch.load(
        f"/home/elhanan/PROJECTS/SHEBA_HABERMAN_YT/temporal_transformer/data/transformer__metadata_weight_idx__{NUM_YEARS}y.pt").to(device)
    metadata_idx = torch.load(
        f"/home/elhanan/PROJECTS/SHEBA_HABERMAN_YT/temporal_transformer/data/transformer__metadata_idx__{NUM_YEARS}y.pt").to(device)
    with open(f'/home/elhanan/PROJECTS/SHEBA_HABERMAN_YT/temporal_transformer/data/event_type_vocab__{NUM_YEARS}y.pkl',
              'rb') as f:
        event_type_vocab = pickle.load(f)
    with open(f'/home/elhanan/PROJECTS/SHEBA_HABERMAN_YT/temporal_transformer/data/cat_value_vocab__{NUM_YEARS}y.pkl',
              'rb') as f:
        cat_value_vocab = pickle.load(f)
    labels = torch.load(
        f"/home/elhanan/PROJECTS/SHEBA_HABERMAN_YT/temporal_transformer/data/transformer__labels__{NUM_YEARS}y.pt").to(device)

    return TransformerBatch(
        t_values=t_values,
        src_mask=src_mask,
        event_idx=event_idx,
        value_idx=value_idx,
        numeric_value=numeric_value,
        value_type_mask=value_type_mask,
        metadata_weight_idx=metadata_weight_idx,
        metadata_idx=metadata_idx,
        event_type_vocab=event_type_vocab,
        cat_value_vocab=cat_value_vocab,
        labels=labels
    )


def train_test_outer_split(transformer_batch: TransformerBatch = None, NUM_YEARS: int = 2):
    # === Outer split (train/test) ===
    idx = torch.arange(transformer_batch.event_idx.size(0))
    labels_np = transformer_batch.labels.cpu().numpy()
    train_idx, test_idx = train_test_split(
        idx, train_size=1600, stratify=labels_np, random_state=42
    )
    torch.save(train_idx,
               f"/home/elhanan/PROJECTS/SHEBA_HABERMAN_YT/temporal_transformer/data/transformer__train_idx__{NUM_YEARS}y.pt")
    torch.save(test_idx,
               f"/home/elhanan/PROJECTS/SHEBA_HABERMAN_YT/temporal_transformer/data/transformer__test_idx__{NUM_YEARS}y.pt")

    return train_idx, test_idx

def run_model_selection(param_grid, train_idx, transformer_batch: TransformerBatch = None, NUM_YEARS: int = 2):

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    labels_np = transformer_batch.labels.cpu().numpy()

    # Name each combo
    param_grid_combination = {
        f"combo_{i}": combo for i, combo in enumerate(combinations)
    }
    # sub = {k: param_grid_combination[k] for k in ("combo_0", "combo_1", "combo_2", "combo_3")} FOR TOY RUN
    # Now pass the full dict to grid_param_check
    results_dict, best_cfgs = grid_param_check(
        param_grid_combination,
        train_idx, labels_np,
        transformer_batch.event_idx, transformer_batch.value_idx, transformer_batch.numeric_value, transformer_batch.value_type_mask,
        transformer_batch.t_values, transformer_batch.src_mask,
        None,  # not used since metadata_data is unpacked later
        transformer_batch.labels,
        transformer_batch.event_type_vocab, transformer_batch.cat_value_vocab,
        n_bootstraps=50, n_splits=4, max_epochs=50, patience=10,
        results_dir="./results"
    )

    return results_dict, best_cfgs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_batch=load_data(NUM_YEARS=2, device=device)
    train_idx, test_idx = train_test_outer_split(transformer_batch, NUM_YEARS=2)
    param_grid = {"capacity": [{"lr": 1e-4, "dropout": 0.3,
                                "num_heads": 1, "d_ff": 128, "num_layers": 1, "cls_hidden": 64, "weight_decay": 1e-5},
                               {"lr": 1e-4, "dropout": 0.3,
                                "num_heads": 1, "d_ff": 256, "num_layers": 2, "cls_hidden": 64, "weight_decay": 1e-5}],
                  "pooling": ["max", "mean"],
                  "initial_emb_dim": [32, 64],
                  "event_emb_agg_method": [EventEmbAggregateMethod.SUM],
                  "event_time_emb_agg_method": [EventTimeEmbAggregateMethod.SUM, EventTimeEmbAggregateMethod.CONCAT],
                  "time_emb_type": [TimeEmbeddingType.REL_POS_ENC, TimeEmbeddingType.CVE, TimeEmbeddingType.TIME2VEC],
                  "md_token": [True, False],
                  "metadata_size_and_data": [(5, transformer_batch.metadata_idx, "metadata_idx"),
                                             (7, transformer_batch.metadata_weight_idx, "metadata_weight_idx")], }

    results_dict, best_cfgs = run_model_selection(param_grid, train_idx, transformer_batch, NUM_YEARS=2)

if __name__ == "__main__":
    main()