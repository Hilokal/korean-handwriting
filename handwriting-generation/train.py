import math
import os
import time

import torch
import torch.nn as nn
from handwriting_dataset import HandwritingData, HandwritingDataset
from model import HandwritingRNN
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split


def mdn_loss(
    pi_logits: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    rho: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood of `target` under a mixture of bivariate Gaussians.

    Shapes: pi_logits (B,S,K), mu (B,S,K,2), log_sigma (B,S,K,2), rho (B,S,K),
    target (B,S,2) [true dx,dy], mask (B,S) bool (True = real, not padding).

    Computed in log-space (logsumexp) so tight Gaussians / far targets don't
    underflow to log(0) = -inf.
    """
    x = target[..., 0:1]  # (B,S,1) -> broadcasts over K
    y = target[..., 1:2]
    mu_x, mu_y = mu[..., 0], mu[..., 1]  # (B,S,K)
    log_sx, log_sy = log_sigma[..., 0], log_sigma[..., 1]
    sig_x, sig_y = log_sx.exp(), log_sy.exp()

    nx = (x - mu_x) / sig_x  # standardized residuals (B,S,K)
    ny = (y - mu_y) / sig_y
    z = nx**2 + ny**2 - 2 * rho * nx * ny
    omr2 = torch.clamp(1 - rho**2, min=1e-6)  # 1 - rho^2

    # log of each component's bivariate-normal density
    log_n = (
        -z / (2 * omr2)
        - math.log(2 * math.pi)
        - log_sx
        - log_sy
        - 0.5 * torch.log(omr2)
    )  # (B,S,K)

    log_pi = torch.log_softmax(pi_logits, dim=-1)  # (B,S,K)
    log_p = torch.logsumexp(log_pi + log_n, dim=-1)  # (B,S) log mixture likelihood

    return -(log_p * mask).sum() / mask.sum()


def collate_fn(batch: list[HandwritingData]):
    """Pad stroke and character sequences to common lengths within a batch."""
    # Filter out empty sequences
    filtered = [x for x in batch if len(x["strokes"]) > 1]
    if not filtered:
        return None, None, None, None, None, None

    # Sort by length (descending) for potential pack_padded_sequence use
    filtered.sort(key=lambda x: len(x["strokes"]), reverse=True)
    lengths = torch.tensor([len(x["strokes"]) for x in filtered])

    strokes = [x["strokes"] for x in filtered]

    # Pad stroke sequences over time
    padded = pad_sequence(strokes, batch_first=True, padding_value=0)

    # Create input (all but last) and target (all but first)
    inputs = padded[:, :-1, :]
    targets = padded[:, 1:, :]

    # Pad the per-sample character (jamo-triple) sequences to a common length and
    # build a mask so the attention window ignores padding characters.
    token_seqs = [x["tokens"] for x in filtered]  # each (U_i, 3)
    u_lengths = torch.tensor([t.size(0) for t in token_seqs])
    tokens = pad_sequence(token_seqs, batch_first=True, padding_value=0)  # (B, U, 3)
    token_mask = torch.arange(tokens.size(1))[None, :] < u_lengths[:, None]  # (B, U)

    return tokens, token_mask, inputs, targets, lengths - 1, padded


def compute_losses(
    model: HandwritingRNN,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    pen_criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass + (xy_loss, pen_loss), masked to real (non-padding) steps."""
    mdn_raw, pen_out, _, _ = model.forward(tokens, inputs, token_mask)
    pi_logits, mu, log_sigma, rho = model.mdn_params(mdn_raw)

    batch_size, seq_len = inputs.shape[0], inputs.shape[1]
    mask = (
        torch.arange(seq_len, device=inputs.device)[None, :]
        < lengths.to(inputs.device)[:, None]
    )  # (B, S)

    # MDN negative log-likelihood over (dx, dy)
    xy_loss = mdn_loss(pi_logits, mu, log_sigma, rho, targets[:, :, :2], mask)

    # Binary end-of-stroke loss (masked BCE; pen_criterion has reduction='none')
    pen_bce = pen_criterion(pen_out.squeeze(-1), targets[:, :, 2])  # (B, S)
    pen_loss = (pen_bce * mask).sum() / mask.sum()

    return xy_loss, pen_loss


def train_epoch(
    model: HandwritingRNN,
    dataloader: DataLoader[HandwritingData],
    optimizer: torch.optim.Optimizer,
    pen_criterion: nn.Module,
    device: torch.device | str,
):
    _ = model.train()
    total_loss = 0
    total_xy_loss = 0
    total_pen_loss = 0
    num_batches = 0

    for batch in dataloader:
        tokens, token_mask, inputs, targets, lengths, _ = batch
        if inputs is None:
            continue

        tokens = tokens.to(device)
        token_mask = token_mask.to(device)
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        xy_loss, pen_loss = compute_losses(
            model, tokens, token_mask, inputs, targets, lengths, pen_criterion
        )
        loss = xy_loss + pen_loss

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_xy_loss += xy_loss.item()
        total_pen_loss += pen_loss.item()
        num_batches += 1

    if num_batches == 0:
        return 0, 0, 0

    return (
        total_loss / num_batches,
        total_xy_loss / num_batches,
        total_pen_loss / num_batches,
    )


@torch.no_grad()
def evaluate(
    model: HandwritingRNN,
    dataloader: DataLoader[HandwritingData],
    pen_criterion: nn.Module,
    device: torch.device | str,
) -> tuple[float, float, float]:
    _ = model.eval()
    total_loss = 0
    total_xy_loss = 0
    total_pen_loss = 0
    num_batches = 0

    for batch in dataloader:
        tokens, token_mask, inputs, targets, lengths, _ = batch
        tokens: torch.Tensor
        inputs: torch.Tensor | None
        targets: torch.Tensor

        if inputs is None:
            continue

        tokens = tokens.to(device)
        token_mask = token_mask.to(device)
        inputs = inputs.to(device)
        targets = targets.to(device)

        xy_loss, pen_loss = compute_losses(
            model, tokens, token_mask, inputs, targets, lengths, pen_criterion
        )
        loss = xy_loss + pen_loss

        total_loss += loss.item()
        total_xy_loss += xy_loss.item()
        total_pen_loss += pen_loss.item()
        num_batches += 1

    if num_batches == 0:
        return 0, 0, 0

    return (
        total_loss / num_batches,
        total_xy_loss / num_batches,
        total_pen_loss / num_batches,
    )


def main():
    # Hyperparameters
    # Capacity knobs, env-overridable (must match at inference -- see load_model).
    hidden_size = int(os.environ.get("HIDDEN_SIZE", "128"))
    num_layers = int(os.environ.get("NUM_LAYERS", "2"))
    embedding_size = int(os.environ.get("EMBEDDING_SIZE", "3"))
    dropout = 0.2
    batch_size = int(os.environ.get("BATCH_SIZE", "64"))
    learning_rate = float(os.environ.get("LR", "1e-3"))
    # High cap so early stopping (below) is the real terminator -- the point is to
    # let the model fully converge, not stop at an arbitrary epoch count.
    num_epochs = int(os.environ.get("NUM_EPOCHS", "6000"))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Dataset and dataloader.
    #   EXPORT=1    -> real multi-character lines from the production export.
    #   SYNTHETIC=1 -> stitched multi-character samples (window smoke test).
    #   (neither)   -> the original single-character folders.
    if os.environ.get("EXPORT"):
        from handwriting_dataset import ExportDataset, split_by_text
        from torch.utils.data import Subset

        dataset = ExportDataset()
        print(f"Using production export dataset ({len(dataset)} recordings)")
        train_idx, val_idx = split_by_text(dataset, val_frac=0.2)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
    elif os.environ.get("SYNTHETIC"):
        from synthetic_dataset import SyntheticMultiCharDataset

        dataset = SyntheticMultiCharDataset()
        print(f"Using synthetic multi-character dataset ({len(dataset)} samples)")
        train_size = int(0.8 * len(dataset))
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, len(dataset) - train_size],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        dataset = HandwritingDataset()
        print(f"Dataset size: {len(dataset)} samples")
        train_size = int(0.8 * len(dataset))
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, len(dataset) - train_size],
            generator=torch.Generator().manual_seed(42),
        )
    print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Model
    model = HandwritingRNN(
        input_size=3,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        embedding_size=embedding_size,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss functions. XY uses the MDN NLL (see compute_losses); pen is a masked
    # binary cross-entropy over the end-of-stroke flag. pos_weight upweights the
    # rare positive (end-of-stroke) class (~1 per stroke).
    pen_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(10.0).to(device), reduction="none"
    )

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=30
    )

    # Training loop bookkeeping.
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 100  # Stop if no improvement for this many epochs
    start_epoch = 0

    # RESUME=<checkpoint.pt> continues a run exactly (model + optimizer + scheduler
    # + epoch + best-so-far). INIT_WEIGHTS=<best_model.pt> warm-starts from weights
    # only (fresh optimizer/scheduler) -- for continuing a model whose run predates
    # full-state checkpoints.
    resume_path = os.environ.get("RESUME")
    init_path = os.environ.get("INIT_WEIGHTS")
    if resume_path:
        # Our own trusted checkpoint; contains optimizer/scheduler state, not just
        # tensors, so weights_only must be False (the torch>=2.6 default is True).
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_val_loss = ck["best_val_loss"]
        patience_counter = ck["patience_counter"]
        print(
            f"Resumed from {resume_path}: epoch {start_epoch}, best val {best_val_loss:.4f}"
        )
    elif init_path:
        model.load_state_dict(
            torch.load(init_path, map_location=device, weights_only=True)
        )
        # Baseline against the loaded model so early stopping measures real progress.
        best_val_loss, _, _ = evaluate(model, val_loader, pen_criterion, device)
        torch.save(model.state_dict(), "best_model.pt")
        print(f"Warm-started from {init_path}: initial val loss {best_val_loss:.4f}")

    start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        train_loss, train_xy, train_pen = train_epoch(
            model, train_loader, optimizer, pen_criterion, device
        )

        val_loss, val_xy, val_pen = evaluate(
            model, val_loader, pen_criterion, device
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1

        # Full-state checkpoint every epoch so the run can resume after any stop.
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
            },
            "checkpoint.pt",
        )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            epochs_done = epoch + 1
            epochs_remaining = num_epochs - epochs_done
            # Rate is over epochs done THIS session (a resume starts mid-count).
            session_done = epoch - start_epoch + 1
            eta = (elapsed / session_done) * epochs_remaining

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))

            print(
                f"Epoch {epochs_done:3d}/{num_epochs} | "
                + f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                + f"XY: {val_xy:.4f} | Pen: {val_pen:.4f} | "
                + f"Elapsed: {elapsed_str} | ETA: {eta_str}"
            )

        if patience_counter >= early_stop_patience:
            print(
                f"\nEarly stopping at epoch {epoch + 1} (no improvement for {early_stop_patience} epochs)"
            )
            break

    total_time = time.time() - start_time
    total_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
    print(f"\nTraining complete in {total_str}. Best val loss: {best_val_loss:.4f}")
    print("Model saved to best_model.pt")


if __name__ == "__main__":
    main()
