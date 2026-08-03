# Defer annotation evaluation so class-body forward references (e.g. the
# `gru: GRUWithSlidingAttention` hint above its class definition) don't raise on
# Python < 3.14, where annotations are evaluated eagerly. Keeps the code portable.
from __future__ import annotations

import torch
import torch.nn as nn

JamoCount = 3

from tokenizer import LeadingCount, SymbolCount, TrailingCount, VowelCount, tokenize


class HandwritingRNN(nn.Module):
    """Simple GRU-based RNN for handwriting generation.

    Predicts the next point (x, y, penState) given a sequence of previous points.
    """

    gru: GRUWithSlidingAttention

    mdn_head: nn.Linear
    pen_head: nn.Linear

    leading_embeddings: nn.Embedding
    vowel_embeddings: nn.Embedding
    trailing_embeddings: nn.Embedding

    def __init__(
        self,
        input_size=3,
        hidden_size=256,
        num_layers=2,
        dropout=0.1,
        num_mixtures=20,
        sliding_window_k=10,
        abs_pos_dim=2,
        embedding_size=3,
    ):
        super().__init__()

        self.num_mixtures = num_mixtures

        # Absolute-position input channels appended to the deltas: 2 = (x, y),
        # 1 = x only, 0 = off (delta-only, the original model). Deltas give
        # translation-invariant *shape*; absolute position gives *placement* --
        # where the pen is along the line, which the window's kappa only knows in
        # logical (character-index) terms. Standardized to the deltas' scale.
        self.abs_pos_dim = abs_pos_dim
        self.register_buffer(
            "pos_mean", torch.tensor([15.378, 0.879]).view(1, 1, 2)
        )
        self.register_buffer("pos_std", torch.tensor([9.931, 0.620]).view(1, 1, 2))

        self.leading_embeddings = nn.Embedding(LeadingCount, embedding_size)
        self.vowel_embeddings = nn.Embedding(VowelCount, embedding_size)
        self.trailing_embeddings = nn.Embedding(TrailingCount, embedding_size)

        # Non-Hangul units (space, punctuation) condition on this instead of the
        # jamo tables. Emits the same width as the three jamo embeddings concatenated
        # (embedding_size * JamoCount) so the window vector stays one fixed size.
        self.symbol_embeddings = nn.Embedding(SymbolCount, embedding_size * JamoCount)

        window_dim = embedding_size * JamoCount  # size of c_u / the window vector
        self.gru = GRUWithSlidingAttention(
            input_size=input_size + abs_pos_dim,
            window_dim=window_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            sliding_window_k=sliding_window_k,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output heads. With Graves' skip connections from every hidden layer to
        # the output, the heads read all layers' hidden states concatenated, so
        # their input width is hidden_size * num_layers.
        head_in = hidden_size * num_layers
        # MDN head: a mixture of K bivariate Gaussians over the next (dx, dy).
        # Each component needs 6 params: pi(1) + mu(2) + sigma(2) + rho(1).
        self.mdn_head = nn.Linear(head_in, num_mixtures * 6)

        # Binary end-of-stroke head (1 logit -> sigmoid). Reading every layer lets
        # it see the stroke-end signal, which lives in layer 1 (not the top layer).
        self.pen_head = nn.Linear(head_in, 1)

        # Start sigma near data scale so densities aren't absurd early on.
        # Bias layout matches the split order: [pi, mu, log_sigma, rho].
        K = num_mixtures
        with torch.no_grad():
            self.mdn_head.bias[3 * K : 5 * K].fill_(0.0)  # log_sigma -> sigma = 1

    def mdn_params(self, raw: torch.Tensor):
        """Split a raw MDN head output into distribution parameters.

        Args:
            raw: (..., 6K) output of mdn_head

        Returns:
            pi_logits: (..., K)     mixture weights (pre-softmax)
            mu:        (..., K, 2)  per-component means of (dx, dy)
            log_sigma: (..., K, 2)  per-component log std (exp -> positive sigma)
            rho:       (..., K)     per-component x-y correlation in (-1, 1)
        """
        K = self.num_mixtures
        pi_logits, mu, log_sigma, rho_raw = torch.split(
            raw, [K, 2 * K, 2 * K, K], dim=-1
        )
        mu = mu.reshape(*mu.shape[:-1], K, 2)
        log_sigma = log_sigma.reshape(*log_sigma.shape[:-1], K, 2)
        rho = torch.tanh(rho_raw)
        return pi_logits, mu, log_sigma, rho

    def forward(
        self,
        tokens: torch.Tensor,
        x: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple, torch.Tensor]:
        """
        Args:
            tokens: Long Tensor of shape (batch, U, 4) -- per unit,
                    [leading, vowel, trailing, symbol]. symbol 0 = Hangul syllable
                    (use the jamo slots); symbol >= 1 = a space/punctuation token.
                    U is the (padded) number of units.
            x: Input tensor of shape (batch, seq_len, 3) containing [dx, dy, penState]
            token_mask: (batch, U) bool, True = real character, False = padding.
                        None treats every character as real.
            state: Optional (hidden, w, kappa) from a previous forward pass, for
                   step-by-step generation.

        Returns:
            mdn_raw: Raw MDN head output (batch, seq_len, 6*num_mixtures).
                     Use mdn_params() to split into (pi, mu, sigma, rho).
            pen_out: Predicted end-of-stroke logits (batch, seq_len, 1)
            state:   ((hidden, w, kappa), abs_pos) -- GRU/window state plus the
                     running absolute position, threaded across generation steps
            phi:     (batch, seq_len, U) attention weights over characters
        """
        # Split the combined state into the GRU/window state and the running
        # absolute position carried across step-by-step generation calls.
        if state is None:
            gru_state, pos0 = None, x.new_zeros(x.size(0), 2)
        else:
            gru_state, pos0 = state

        # Absolute position = start offset + cumulative deltas, standardized to the
        # deltas' scale. Appended to the input as the placement signal the
        # delta-only model lacked (abs_pos_dim channels: (x, y), x only, or none).
        abs_pos = pos0.unsqueeze(1) + torch.cumsum(x[:, :, :2], dim=1)  # (B, S, 2)
        norm_pos = (abs_pos - self.pos_mean) / self.pos_std
        aug_x = torch.cat((x, norm_pos[:, :, : self.abs_pos_dim]), dim=-1)

        # Build one conditioning vector per unit -- the sequence the window
        # attends over. Hangul units use the concat of their three jamo embeddings
        # (9-dim); space/punctuation units use the symbol embedding (same 9-dim) so
        # their strokes attend to their own token instead of a neighbouring jamo.
        leading = self.leading_embeddings(tokens[:, :, 0])  # (B, U, embedding_size)
        vowel = self.vowel_embeddings(tokens[:, :, 1])
        trailing = self.trailing_embeddings(tokens[:, :, 2])
        c_jamo = torch.cat((leading, vowel, trailing), dim=-1)  # (B, U, 3*embedding_size)

        symbol = tokens[:, :, 3]  # (B, U); 0 = Hangul, >= 1 = symbol
        c_symbol = self.symbol_embeddings(symbol)  # (B, U, 3*embedding_size)
        c = torch.where((symbol > 0).unsqueeze(-1), c_symbol, c_jamo)

        output, gru_state, phi = self.gru(aug_x, c, token_mask, gru_state)

        # Predict next point: a mixture distribution over (dx, dy), and a
        # binary end-of-stroke logit.
        mdn_raw = self.mdn_head(output)
        pen_out = self.pen_head(output)  # (batch, seq_len, 1)

        new_pos = pos0 + x[:, :, :2].sum(dim=1)  # ending absolute position
        return mdn_raw, pen_out, (gru_state, new_pos), phi

    def generate(
        self,
        start_seq: torch.Tensor,
        character: str,
        max_len: int = 500,
        temperature: float = 1.0,
        bias: float = 0.0,
        num_strokes: int | None = None,
        device: torch.device | str = "cpu",
    ):
        """Generate a handwriting sequence autoregressively.

        Args:
            start_seq: Starting sequence tensor of shape (1, seq_len, 3)
            character: Korean hanguel character to generate
            max_len: Maximum number of points to generate
            temperature: Sampling temperature for the end-of-stroke sigmoid
            bias: MDN sampling bias (Graves). Higher = tighter sigma + sharper
                  mixture = cleaner/less varied strokes; 0 = unbiased.
            num_strokes: If set, stop after this many strokes are drawn (the
                  character's expected stroke count). This is how the drawing
                  terminates — there is no learned end-of-sequence signal. If
                  None, generate until max_len.
            device: Device to run on

        Returns:
            Generated sequence tensor of shape (1, total_len, 3)
        """
        _ = self.eval()
        generated = start_seq.clone().to(device)
        state = None
        strokes_done = 0  # number of completed strokes (end-of-stroke emitted)

        tokens = tokenize(character).unsqueeze(0).to(device)  # (1, U, 3)
        U = tokens.size(1)
        token_mask = torch.ones(1, U, device=device)

        with torch.no_grad():
            # Process the starting sequence to prime the recurrent + window state
            mdn_raw, pen_out, state, phi = self(tokens, generated, token_mask, state)

            for _ in range(max_len):
                # --- end-of-stroke: binary, greedy if temperature=0 else sample ---
                pen_logit = pen_out[:, -1, :]  # (1, 1)
                if temperature == 0:
                    pen_state = (torch.sigmoid(pen_logit) > 0.5).float()
                else:
                    p = torch.sigmoid(pen_logit / temperature)
                    pen_state = torch.bernoulli(p)  # (1, 1)

                # --- offset (dx, dy): sample from the mixture ---
                pi_logits, mu, log_sigma, rho = self.mdn_params(mdn_raw[:, -1])
                # (1, K), (1, K, 2), (1, K, 2), (1, K)
                pi = torch.softmax(pi_logits * (1 + bias), dim=-1)
                sigma = torch.exp(log_sigma - bias)

                k = torch.multinomial(pi, 1).item()  # pick a component
                mx, my = mu[0, k]
                sx, sy = sigma[0, k]
                r = rho[0, k]

                # Sample the bivariate Gaussian via the Cholesky factor of its cov.
                z1, z2 = torch.randn(2, device=device)
                dx = mx + sx * z1
                dy = my + r * sy * z1 + sy * torch.sqrt(1 - r**2) * z2

                last_xy = torch.stack([dx, dy]).reshape(1, 1, 2)

                # Combine into next input ([dx, dy, end_of_stroke])
                next_point = torch.cat([last_xy, pen_state.unsqueeze(-1)], dim=-1)
                generated = torch.cat([generated, next_point], dim=1)

                # Terminate after the character's expected number of strokes.
                if pen_state.item() == 1:
                    strokes_done += 1
                    if num_strokes is not None and strokes_done >= num_strokes:
                        break

                # Get next prediction (thread the window state, not just hidden)
                mdn_raw, pen_out, state, phi = self(
                    tokens, next_point, token_mask, state
                )

                # Multi-character termination: stop once the attention window has
                # slid past the last character. Approximates Graves' phi(t, U+1) >
                # phi(t, u) test via the window centre kappa. Used only when
                # stroke-count termination isn't requested (single-char inference
                # passes num_strokes, so this branch is skipped there).
                # kappa lives in the GRU/window part of the combined state.
                kappa = state[0][2]
                if num_strokes is None and U > 1 and kappa.mean().item() >= U:
                    break

        return generated


class GRUWithSlidingAttention(nn.Module):
    """Stacked-GRUCell loop with a Graves-style attention window and the synthesis
    network's full skip connectivity.

    Runs the GRU one timestep at a time (rather than nn.GRU) so the first layer's
    hidden state can drive a soft attention window over the character sequence and
    the window vector ``w_t`` can be fed back into the recurrence.

    Window (Graves 2013, eqs. 46-51), computed from the first hidden layer h1_t:
        (a_hat, b_hat, k_hat) = affine(h1_t)                  # each (B, K)
        alpha = exp(a_hat)                                    # amplitudes  > 0
        beta  = exp(b_hat)                                    # widths      > 0
        kappa_t = kappa_{t-1} + exp(k_hat)                    # locations, monotonic
        phi(t, u) = sum_k alpha_k * exp(-beta_k (kappa_k - u)^2)
        w_t = sum_u phi(t, u) * c_u

    Skip connectivity, matching Graves' synthesis network:
      - **Input skip:** the raw input x_t feeds *every* layer.
      - **Window skip:** w feeds every layer -- layer 0 gets w_{t-1} (w_t is
        computed from h1_t, so feeding it back to layer 0 at the same step would
        make a cycle); higher layers get the current w_t.
      - **Output skip:** the heads read *every* layer's hidden state (this loop
        returns their concatenation), not just the top layer -- the stroke-end
        signal lives in layer 1, so the pen head needs access to it.
    """

    alpha_head: nn.Linear
    beta_head: nn.Linear
    kappa_head: nn.Linear

    def __init__(
        self,
        input_size,
        window_dim,
        hidden_size,
        num_layers,
        dropout,
        sliding_window_k,
    ):
        super().__init__()

        if num_layers < 2:
            raise Exception("num_layers must be at least 2")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sliding_window_k = sliding_window_k
        self.window_dim = window_dim

        # Layer 0 input is [x_t, w]; higher layers also get the input skip and the
        # layer below: [x_t, h_below, w].
        self.layers = nn.ModuleList(
            [
                nn.GRUCell(
                    input_size + window_dim
                    if i == 0
                    else input_size + hidden_size + window_dim,
                    hidden_size,
                )
                for i in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        self.alpha_head = nn.Linear(hidden_size, sliding_window_k)
        self.beta_head = nn.Linear(hidden_size, sliding_window_k)
        self.kappa_head = nn.Linear(hidden_size, sliding_window_k)

        # Bias kappa so it advances ~1 character index per ~e^4 ~= 55 points,
        # matching the measured ~58 points/character. Without this the default
        # exp(0)=1 step slides the window off the end of the string in ~3 steps,
        # so training would start with attention pointing past the text.
        with torch.no_grad():
            self.kappa_head.bias.fill_(-4.0)

    def forward(self, x_seq, c, c_mask=None, state=None):
        """
        Args:
            x_seq:  (B, S, x_dim) pen input [dx, dy, penState].
            c:      (B, U, window_dim) per-character conditioning vectors.
            c_mask: (B, U) True/1 = real character, 0 = padding. None = all real.
            state:  optional (hidden, w, kappa) carried from a previous call, so
                    generation can step one point at a time (kappa and w must
                    persist across calls or the window resets to the start).

        Returns:
            output: (B, S, hidden_size * num_layers) -- every layer's hidden state
                    concatenated (the output skip connection), fed to the heads.
            state:  (hidden, w, kappa) final recurrent + window state.
            phi:    (B, S, U) attention weight per character per step (for
                    monitoring the window slide and for termination).
        """
        B = x_seq.size(0)
        U = c.size(1)
        device = x_seq.device

        if c_mask is None:
            c_mask = torch.ones(B, U, device=device)
        c_mask = c_mask.to(c.dtype)

        if state is None:
            hidden = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
            w = torch.zeros(B, self.window_dim, device=device)
            kappa = torch.zeros(B, self.sliding_window_k, device=device)
        else:
            hidden, w, kappa = state

        u = torch.arange(U, device=device).view(1, 1, U)  # (1, 1, U)

        outputs = []
        phis = []

        for t in range(x_seq.size(1)):
            x_t = x_seq[:, t]
            new_hidden = []

            # Layer 0: input point + the PREVIOUS step's window (w_{t-1}).
            h = self.layers[0](torch.cat([x_t, w], dim=-1), hidden[0])
            new_hidden.append(h)  # clean state (never dropout-contaminated)

            # Sliding attention window, driven by the first hidden layer.
            alpha = torch.exp(self.alpha_head(h)).unsqueeze(-1)  # (B, K, 1)
            beta = torch.exp(self.beta_head(h)).unsqueeze(-1)  # (B, K, 1)
            kappa = kappa + torch.exp(self.kappa_head(h))  # (B, K), monotonic
            phi = (alpha * torch.exp(-beta * (kappa.unsqueeze(-1) - u) ** 2)).sum(
                1
            )  # (B, U)
            phi = phi * c_mask  # drop padding characters (no renormalisation, per Graves)
            w = torch.bmm(phi.unsqueeze(1), c).squeeze(1)  # w_t (current window)

            # Higher layers: input skip (x_t) + layer below + current window (w_t).
            # Dropout on the vertical path only, so the recurrent state stays clean.
            below = h
            for i in range(1, self.num_layers):
                layer_in = torch.cat([x_t, self.dropout(below), w], dim=-1)
                h = self.layers[i](layer_in, hidden[i])
                new_hidden.append(h)
                below = h

            # Output skip: the heads read every layer's hidden state.
            outputs.append(torch.cat(new_hidden, dim=-1))  # (B, hidden * num_layers)
            phis.append(phi)
            hidden = torch.stack(new_hidden, dim=0)

        output = torch.stack(outputs, dim=1)  # (B, S, hidden * num_layers)
        phi_seq = torch.stack(phis, dim=1)  # (B, S, U)
        return output, (hidden, w, kappa), phi_seq
