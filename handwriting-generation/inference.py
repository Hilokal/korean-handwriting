import argparse
import json
import os

import torch
from handwriting_dataset import FORCE_MEAN, FORCE_STD, HandwritingDataset
from model import HandwritingRNN


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path, device):
    # Must match how the checkpoint was trained (see train.py). Override via
    # HIDDEN_SIZE / NUM_LAYERS to load a wider or deeper checkpoint.
    model = HandwritingRNN(
        input_size=4,
        hidden_size=int(os.environ.get("HIDDEN_SIZE", "128")),
        num_layers=int(os.environ.get("NUM_LAYERS", "2")),
        dropout=0.0,
        embedding_size=int(os.environ.get("EMBEDDING_SIZE", "3")),
        onehot=bool(os.environ.get("ONEHOT")),
    ).to(device)

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def generate_sequence(
    model: HandwritingRNN,
    character: str,
    device,
    seed_seq=None,
    max_len=500,
    temperature=1.0,
    seed=None,
    bias=0.0,
    num_strokes=None,
):
    """Generate a handwriting sequence.

    Args:
        model: Trained model
        device: Device to run on
        seed_seq: Optional starting sequence tensor (1, seq_len, 4)
                  If None, starts with a single point at origin
        max_len: Maximum points to generate
        temperature: Sampling temperature (higher = more random)
        seed: Optional RNG seed. When set, makes the stochastic pen-state
              sampling reproducible (same seed -> same strokes).
        bias: MDN sampling bias. Higher = tighter/cleaner, less varied strokes.

    Returns:
        Generated sequence as numpy array of shape (seq_len, 4)
    """
    if seed is not None:
        torch.manual_seed(seed)

    if seed_seq is None:
        # Start with a single point at origin. penState=0 (mid-stroke) matches
        # the training convention where the first point is not an end-of-stroke;
        # f=0 is the corpus-mean pen force in standardized units.
        seed_seq = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)

    seed_seq = seed_seq.to(device)
    generated = model.generate(
        seed_seq,
        character=character,
        max_len=max_len,
        temperature=temperature,
        bias=bias,
        num_strokes=num_strokes,
        device=device,
    )

    return generated.squeeze(0).cpu().numpy()


def sequence_to_json(sequence):
    """Convert sequence array to JSON-serializable format.

    Pressure is de-standardized back to raw pen force units so generated.json
    matches the recordings' scale (render.py normalizes per-drawing anyway).
    """
    dots = []
    for point in sequence:
        dot = {
            "x": float(point[0]),
            "y": float(point[1]),
            "penState": int(point[2]),
        }
        if len(point) > 3:
            dot["f"] = float(point[3]) * FORCE_STD + FORCE_MEAN
        dots.append(dot)
    return {"dots": dots}


def main():
    parser = argparse.ArgumentParser(description="Generate handwriting sequences")
    _ = parser.add_argument(
        "--model",
        type=str,
        default="best_model.pt",
        help="Path to model checkpoint",
    )
    _ = parser.add_argument("--character", type=str, default="안")
    _ = parser.add_argument(
        "--output",
        type=str,
        default="generated.json",
        help="Output JSON file path",
    )
    _ = parser.add_argument(
        "--max-len",
        type=int,
        default=500,
        help="Maximum sequence length to generate",
    )
    _ = parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more random)",
    )
    _ = parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible sampling (same seed -> same strokes)",
    )
    _ = parser.add_argument(
        "--bias",
        type=float,
        default=0.0,
        help="MDN sampling bias (higher = cleaner/less varied strokes)",
    )
    _ = parser.add_argument(
        "--num-strokes",
        type=int,
        default=None,
        help="Stop after this many strokes (default: the character's modal count)",
    )
    _ = parser.add_argument(
        "--seed-index",
        type=int,
        default=None,
        help="Use first N points from dataset sample as seed",
    )
    _ = parser.add_argument(
        "--seed-length",
        type=int,
        default=20,
        help="Number of points to use from seed sample",
    )

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model}")
    model = load_model(args.model, device)

    # Stroke count for termination: explicit, else the single character's modal
    # count. Multi-character strings terminate via the window (num_strokes=None).
    # modal_stroke_counts() needs the single-char training data, which may be
    # absent on a fresh clone -- fall back to window termination if so.
    num_strokes = args.num_strokes
    if num_strokes is None and len(args.character) == 1:
        try:
            from handwriting_dataset import modal_stroke_counts

            num_strokes = modal_stroke_counts().get(args.character)
        except Exception:
            pass
    print(f"num_strokes: {num_strokes}")

    # Get seed sequence if requested
    seed_seq = None
    if args.seed_index is not None:
        dataset = HandwritingDataset()
        sample = dataset[args.seed_index]
        seed_len = min(args.seed_length, len(sample))
        seed_seq = sample[:seed_len].unsqueeze(0)
        print(f"Using seed from dataset[{args.seed_index}], first {seed_len} points")

    # Generate
    print(
        f"Generating sequence (max_len={args.max_len}, temperature={args.temperature})"
    )
    sequence = generate_sequence(
        model,
        args.character,
        device,
        seed_seq=seed_seq,
        max_len=args.max_len,
        temperature=args.temperature,
        seed=args.seed,
        bias=args.bias,
        num_strokes=num_strokes,
    )

    print(f"Generated {len(sequence)} points")

    # Save to JSON
    output_data = sequence_to_json(sequence)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
