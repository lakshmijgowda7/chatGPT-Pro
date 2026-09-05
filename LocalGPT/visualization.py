"""
LocalGPT: Visualizations (Step 15: LLM X-Ray Visualizations)
Provides robust, interactive Plotly visualizations with comprehensive safeguards for:
- PCA dimensionality reduction for embeddings (0, 1, 2, or N tokens)
- Zero-variance, singular, and identical vector handling
- Transformer 28-layer architectural stack visualization
- Variable attention matrix shapes (1x1, MHA, GQA) heatmaps
- Intermediate hidden state representation spaces (2D PCA)
- Next-token probability bar charts and candidate rankings
- Autoregressive generation flow timeline diagrams
"""

from typing import List, Optional, Dict, Any
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def plot_embeddings_2d(
    tokens: List[str],
    token_ids: List[int],
    embeddings_matrix: np.ndarray,
) -> go.Figure:
    """
    Reduces high-dimensional token embedding vectors to 2D using scikit-learn PCA
    and generates an interactive Plotly scatter plot with labeled points.
    
    Guaranteed to never crash on 0 tokens, 1 token, identical tokens, or singular matrices.
    """
    n_samples = len(tokens)

    # 1. Empty case
    if n_samples == 0 or embeddings_matrix is None or embeddings_matrix.size == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No token embeddings available to visualize.",
            showarrow=False,
            font=dict(size=14, color="gray"),
        )
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
        return fig

    # Ensure matrix is clean 2D float32 array
    clean_mat = np.nan_to_num(np.atleast_2d(embeddings_matrix).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Single token case
    if n_samples == 1:
        coords_2d = np.array([[0.0, 0.0]])
        var_explained = [100.0, 0.0]
        status_note = "Single token: Placed at origin (PCA requires ≥ 2 tokens for variance calculation)"
    # 3. Two tokens case
    elif n_samples == 2:
        try:
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(clean_mat)
            var_explained = [
                float(pca.explained_variance_ratio_[0]) * 100.0 if len(pca.explained_variance_ratio_) > 0 else 100.0,
                float(pca.explained_variance_ratio_[1]) * 100.0 if len(pca.explained_variance_ratio_) > 1 else 0.0,
            ]
            status_note = "2 tokens: 1 principal axis separates the pair"
        except Exception:
            coords_2d = np.array([[-1.0, 0.0], [1.0, 0.0]])
            var_explained = [100.0, 0.0]
            status_note = "2 tokens: Projected along coordinate axis"
    # 4. Multi-token case (>= 3 tokens)
    else:
        try:
            if np.allclose(clean_mat, clean_mat[0]):
                jitter = np.linspace(-0.5, 0.5, n_samples)
                coords_2d = np.column_stack([jitter, np.zeros(n_samples)])
                var_explained = [100.0, 0.0]
                status_note = "Tokens have identical embeddings: Displayed in linear sequence"
            else:
                pca = PCA(n_components=2)
                coords_2d = pca.fit_transform(clean_mat)
                var_explained = [
                    float(pca.explained_variance_ratio_[0]) * 100.0 if len(pca.explained_variance_ratio_) > 0 else 0.0,
                    float(pca.explained_variance_ratio_[1]) * 100.0 if len(pca.explained_variance_ratio_) > 1 else 0.0,
                ]
                status_note = f"Explained Variance: PC1 = {var_explained[0]:.1f}%, PC2 = {var_explained[1]:.1f}%"
        except Exception:
            coords_2d = np.column_stack([np.linspace(-1, 1, n_samples), np.zeros(n_samples)])
            var_explained = [100.0, 0.0]
            status_note = "Fallback projection applied (PCA numerical exception)"

    x_vals = coords_2d[:, 0]
    y_vals = coords_2d[:, 1]

    display_labels = [
        f"<b>{tok.replace(' ', '␣').replace(chr(10), '↵')}</b>"
        for tok in tokens[:n_samples]
    ]

    customdata = [
        [tok, tid, idx, repr(tok)]
        for idx, (tok, tid) in enumerate(zip(tokens[:n_samples], token_ids[:n_samples]))
    ]

    indices = list(range(n_samples))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers+text",
            text=display_labels,
            textposition="top center",
            textfont=dict(size=12, family="Courier New, monospace"),
            marker=dict(
                size=14,
                color=indices,
                colorscale="Viridis",
                showscale=True if n_samples > 2 else False,
                colorbar=dict(title="Token Index", len=0.7) if n_samples > 2 else None,
                line=dict(width=1.5, color="white"),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>Token:</b> %{customdata[0]}<br>"
                "<b>Raw Repr:</b> %{customdata[3]}<br>"
                "<b>Token ID:</b> %{customdata[1]}<br>"
                "<b>Index:</b> %{customdata[2]}<br>"
                "<b>PC 1:</b> %{x:.4f}<br>"
                "<b>PC 2:</b> %{y:.4f}"
                "<extra></extra>"
            ),
        )
    )

    if n_samples > 1:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                line=dict(color="rgba(150, 150, 150, 0.4)", width=1.5, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=dict(
            text=f"<b>2D PCA Projection of Input Token Embeddings</b><br><sup>{status_note}</sup>",
            x=0.02,
        ),
        xaxis=dict(
            title=f"Principal Component 1 ({var_explained[0]:.1f}% var)",
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)",
            zeroline=True,
            zerolinecolor="rgba(128, 128, 128, 0.4)",
        ),
        yaxis=dict(
            title=f"Principal Component 2 ({var_explained[1]:.1f}% var)",
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)",
            zeroline=True,
            zerolinecolor="rgba(128, 128, 128, 0.4)",
        ),
        showlegend=False,
        hovermode="closest",
        margin=dict(l=40, r=40, t=60, b=40),
        height=520,
    )

    return fig


def plot_architecture_stack(
    num_layers: int,
    selected_layer_num: int,
) -> go.Figure:
    """
    Renders an interactive visual architecture stack of the model showing
    Embeddings -> Layer 1 -> ... -> Layer N -> Output with the selected layer highlighted.
    """
    safe_total = max(1, num_layers)
    safe_selected = max(1, min(selected_layer_num, safe_total))

    block_names = ["Embeddings"] + [f"Layer {i}" for i in range(1, safe_total + 1)] + ["Output (LM Head)"]
    n_blocks = len(block_names)

    colors = []
    text_colors = []
    border_widths = []

    for i in range(n_blocks):
        if i == 0 or i == n_blocks - 1:
            colors.append("#2c3e50")
            text_colors.append("#ecf0f1")
            border_widths.append(1)
        else:
            layer_num = i
            if layer_num == safe_selected:
                colors.append("#00ADB5")  # Highlighted Cyan
                text_colors.append("#ffffff")
                border_widths.append(3)
            else:
                colors.append("#1b263b")  # Dark Slate Blue
                text_colors.append("#cbd5e1")
                border_widths.append(1)

    x_positions = [1] * n_blocks

    hover_texts = []
    for i, name in enumerate(block_names):
        if i == 0:
            hover_texts.append("<b>Input Embeddings</b><br>Shape: (seq_len, 1536)<br>Token IDs → Dense Vectors")
        elif i == n_blocks - 1:
            hover_texts.append("<b>Output LM Head</b><br>RMSNorm + Linear Projection<br>1536 → 151,936 Logits")
        else:
            l_num = i
            star = " ⭐ <b>(Active Selection)</b>" if l_num == safe_selected else ""
            hover_texts.append(
                f"<b>Transformer Block {l_num} of {safe_total}</b>{star}<br>"
                f"• Multi-Head Attention (12 Q / 2 KV Heads)<br>"
                f"• SwiGLU FFN (Intermediate: 8960-D)<br>"
                f"• Pre & Post RMSNorm"
            )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=x_positions,
            y=[name for name in reversed(block_names)],
            orientation="h",
            marker=dict(
                color=list(reversed(colors)),
                line=dict(
                    color="white",
                    width=list(reversed(border_widths)),
                ),
            ),
            hoverinfo="text",
            hovertext=list(reversed(hover_texts)),
            text=[f"  <b>{name}</b>" + (" ⭐" if name == f"Layer {safe_selected}" else "") for name in reversed(block_names)],
            textposition="inside",
            textfont=dict(
                size=11,
                family="Courier New, monospace",
                color=list(reversed(text_colors)),
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Transformer Layer Stack ({safe_total} Layers)</b><br><sup>Layer {safe_selected} currently selected for detailed inspection</sup>",
            x=0.02,
        ),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[0, 1.1],
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=10, family="Courier New, monospace"),
        ),
        hovermode="closest",
        margin=dict(l=10, r=10, t=60, b=20),
        height=max(480, 20 * safe_total + 100),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def plot_attention_heatmap(
    tokens: List[str],
    attention_matrix: np.ndarray,
    layer_num: int = 1,
    head_num: Optional[int] = 1,
    is_average: bool = False,
) -> go.Figure:
    """
    Renders an interactive Plotly heatmap of Multi-Head Self-Attention weights.
    """
    if attention_matrix is None or attention_matrix.size == 0 or len(tokens) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No attention matrix data available.",
            showarrow=False,
            font=dict(size=14, color="gray"),
        )
        fig.update_layout(height=400)
        return fig

    mat_2d = np.nan_to_num(np.atleast_2d(attention_matrix).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    seq_len = min(len(tokens), mat_2d.shape[0], mat_2d.shape[1])

    if seq_len == 0:
        mat_2d = np.array([[1.0]], dtype=np.float32)
        seq_len = 1
        tokens = ["<empty>"]

    mat_2d = mat_2d[:seq_len, :seq_len]

    display_labels = [
        f"[{i}] {tok.replace(' ', '␣').replace(chr(10), '↵')}"
        for i, tok in enumerate(tokens[:seq_len])
    ]

    if is_average or head_num is None:
        title_text = f"<b>Self-Attention Heatmap — Layer {layer_num} (Mean Across All Heads)</b>"
        sub_text = f"Averaged attention distribution across all attention heads for Layer {layer_num}"
    else:
        title_text = f"<b>Self-Attention Heatmap — Layer {layer_num} | Head {head_num}</b>"
        sub_text = f"Real attention weights computed by Attention Head {head_num} in Layer {layer_num}"

    custom_hover = []
    for r_idx in range(seq_len):
        row_hover = []
        for c_idx in range(seq_len):
            q_tok = tokens[r_idx] if r_idx < len(tokens) else f"Token {r_idx}"
            k_tok = tokens[c_idx] if c_idx < len(tokens) else f"Token {c_idx}"
            row_hover.append([
                q_tok,
                k_tok,
                r_idx,
                c_idx,
                repr(q_tok),
                repr(k_tok),
            ])
        custom_hover.append(row_hover)

    show_cell_text = seq_len <= 14
    cell_text = np.round(mat_2d, 3).astype(str) if show_cell_text else None

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=mat_2d,
            x=display_labels,
            y=display_labels,
            text=cell_text,
            texttemplate="%{z:.2f}" if show_cell_text else None,
            textfont=dict(size=10, family="Courier New, monospace"),
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(
                title=dict(text="Attention<br>Weight", side="top"),
                tickformat=".2f",
                len=0.85,
                thickness=18,
            ),
            customdata=custom_hover,
            hovertemplate=(
                "<b>Query (From Token):</b> %{customdata[0]} (Index %{customdata[2]})<br>"
                "<b>Key (To Token):</b> %{customdata[1]} (Index %{customdata[3]})<br>"
                "<b>Raw Repr:</b> %{customdata[4]} → %{customdata[5]}<br>"
                "<b>Attention Weight:</b> %{z:.6f}"
                "<extra></extra>"
            ),
        )
    )

    chart_height = max(460, min(800, 40 * seq_len + 180))

    fig.update_layout(
        title=dict(
            text=f"{title_text}<br><sup>{sub_text}</sup>",
            x=0.02,
        ),
        xaxis=dict(
            title="<b>Key Tokens (Attended-to Context)</b> →",
            tickangle=-45,
            tickfont=dict(size=11, family="Courier New, monospace"),
            side="bottom",
            showgrid=False,
        ),
        yaxis=dict(
            title="← <b>Query Tokens (Attending-from Position)</b>",
            autorange="reversed",
            tickfont=dict(size=11, family="Courier New, monospace"),
            showgrid=False,
        ),
        hovermode="closest",
        margin=dict(l=80, r=60, t=70, b=80),
        height=chart_height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def plot_hidden_states_2d(
    tokens: List[str],
    token_ids: List[int],
    hidden_matrix: np.ndarray,
    layer_label: str = "Layer 1",
    layer_num: int = 1,
) -> go.Figure:
    """
    Reduces the 1536-dimensional hidden state representations of the selected layer
    to 2D using PCA and renders an interactive Plotly scatter plot.
    """
    n_samples = len(tokens)

    if n_samples == 0 or hidden_matrix is None or hidden_matrix.size == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No hidden states available to visualize.",
            showarrow=False,
            font=dict(size=14, color="gray"),
        )
        fig.update_layout(height=400)
        return fig

    clean_mat = np.nan_to_num(np.atleast_2d(hidden_matrix).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if n_samples == 1:
        coords_2d = np.array([[0.0, 0.0]])
        var_explained = [100.0, 0.0]
        status_note = f"Single token hidden state from {layer_label} placed at origin (PCA requires ≥ 2 tokens)"
    elif n_samples == 2:
        try:
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(clean_mat)
            var_explained = [
                float(pca.explained_variance_ratio_[0]) * 100.0 if len(pca.explained_variance_ratio_) > 0 else 100.0,
                float(pca.explained_variance_ratio_[1]) * 100.0 if len(pca.explained_variance_ratio_) > 1 else 0.0,
            ]
            status_note = f"2 tokens: 1 principal axis separates hidden states in {layer_label}"
        except Exception:
            coords_2d = np.array([[-1.0, 0.0], [1.0, 0.0]])
            var_explained = [100.0, 0.0]
            status_note = f"2 tokens: Projected coordinate representation for {layer_label}"
    else:
        try:
            if np.allclose(clean_mat, clean_mat[0]):
                jitter = np.linspace(-0.5, 0.5, n_samples)
                coords_2d = np.column_stack([jitter, np.zeros(n_samples)])
                var_explained = [100.0, 0.0]
                status_note = f"Tokens have identical hidden states in {layer_label}: Displayed in sequence"
            else:
                pca = PCA(n_components=2)
                coords_2d = pca.fit_transform(clean_mat)
                var_explained = [
                    float(pca.explained_variance_ratio_[0]) * 100.0 if len(pca.explained_variance_ratio_) > 0 else 0.0,
                    float(pca.explained_variance_ratio_[1]) * 100.0 if len(pca.explained_variance_ratio_) > 1 else 0.0,
                ]
                status_note = f"{layer_label} Representation Space — Explained Variance: PC1 = {var_explained[0]:.1f}%, PC2 = {var_explained[1]:.1f}%"
        except Exception:
            coords_2d = np.column_stack([np.linspace(-1, 1, n_samples), np.zeros(n_samples)])
            var_explained = [100.0, 0.0]
            status_note = f"Fallback projection applied for {layer_label} (PCA numerical exception)"

    x_vals = coords_2d[:, 0]
    y_vals = coords_2d[:, 1]

    token_norms = [float(np.linalg.norm(clean_mat[i])) for i in range(n_samples)]

    display_labels = [
        f"<b>{tok.replace(' ', '␣').replace(chr(10), '↵')}</b>"
        for tok in tokens[:n_samples]
    ]

    customdata = [
        [tok, tid, idx, repr(tok), round(token_norms[idx], 4)]
        for idx, (tok, tid) in enumerate(zip(tokens[:n_samples], token_ids[:n_samples]))
    ]

    indices = list(range(n_samples))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers+text",
            text=display_labels,
            textposition="top center",
            textfont=dict(size=12, family="Courier New, monospace"),
            marker=dict(
                size=14,
                color=indices,
                colorscale="Plasma",
                showscale=True if n_samples > 2 else False,
                colorbar=dict(title="Token Index", len=0.7) if n_samples > 2 else None,
                line=dict(width=1.5, color="white"),
            ),
            customdata=customdata,
            hovertemplate=(
                f"<b>{layer_label} Hidden State</b><br>"
                "<b>Token:</b> %{customdata[0]}<br>"
                "<b>Raw Repr:</b> %{customdata[3]}<br>"
                "<b>Token ID:</b> %{customdata[1]}<br>"
                "<b>Index:</b> %{customdata[2]}<br>"
                "<b>Vector L2 Norm:</b> %{customdata[4]}<br>"
                "<b>PC 1:</b> %{x:.4f}<br>"
                "<b>PC 2:</b> %{y:.4f}"
                "<extra></extra>"
            ),
        )
    )

    if n_samples > 1:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                line=dict(color="rgba(255, 128, 0, 0.45)", width=1.5, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=dict(
            text=f"<b>2D PCA Projection: {layer_label} Hidden States</b><br><sup>{status_note}</sup>",
            x=0.02,
        ),
        xaxis=dict(
            title=f"Principal Component 1 ({var_explained[0]:.1f}% var)",
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)",
            zeroline=True,
            zerolinecolor="rgba(128, 128, 128, 0.4)",
        ),
        yaxis=dict(
            title=f"Principal Component 2 ({var_explained[1]:.1f}% var)",
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)",
            zeroline=True,
            zerolinecolor="rgba(128, 128, 128, 0.4)",
        ),
        showlegend=False,
        hovermode="closest",
        margin=dict(l=40, r=40, t=60, b=40),
        height=520,
    )

    return fig


def plot_next_token_probabilities(
    top_predictions: List[Dict[str, Any]],
    title_suffix: str = "",
) -> go.Figure:
    """
    Renders an interactive Plotly horizontal bar chart showing the model's top predicted next tokens
    and their exact softmax probabilities.
    """
    if not top_predictions:
        fig = go.Figure()
        fig.add_annotation(
            text="No token predictions available.",
            showarrow=False,
            font=dict(size=14, color="gray"),
        )
        fig.update_layout(height=400)
        return fig

    predictions = list(reversed(top_predictions))

    y_labels = [
        f"<b>[#{item['rank']}]</b> {item['token_display']} <i>(ID: {item['token_id']})</i>"
        for item in predictions
    ]
    x_probs_pct = [item["probability_pct"] for item in predictions]
    text_labels = [item["probability_pct_str"] for item in predictions]

    customdata = [
        [
            item["rank"],
            item["token_display"],
            item["token_id"],
            item["token_repr"],
            item["probability_pct"],
            item["probability"],
            item.get("logit", 0.0),
        ]
        for item in predictions
    ]

    colors = []
    for item in predictions:
        if item["rank"] == 1:
            colors.append("#00ADB5")  # Vibrant Cyan for Top 1
        elif item["rank"] == 2:
            colors.append("#3a86ff")
        elif item["rank"] == 3:
            colors.append("#4361ee")
        else:
            colors.append("#4895ef")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=x_probs_pct,
            y=y_labels,
            orientation="h",
            text=text_labels,
            textposition="auto",
            textfont=dict(size=11, family="Courier New, monospace", color="white"),
            marker=dict(
                color=colors,
                line=dict(color="rgba(255, 255, 255, 0.3)", width=1),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>Rank #%{customdata[0]}:</b> %{customdata[1]}<br>"
                "<b>Token ID:</b> %{customdata[2]}<br>"
                "<b>Raw Repr:</b> %{customdata[3]}<br>"
                "<b>Probability:</b> %{customdata[4]:.4f}% (Raw: %{customdata[5]:.6f})<br>"
                "<b>Logit Score:</b> %{customdata[6]:.4f}"
                "<extra></extra>"
            ),
        )
    )

    max_val = max(x_probs_pct) if x_probs_pct else 100.0
    x_max_limit = min(100.0, max_val * 1.15 if max_val < 85 else 100.0)

    top_1_item = top_predictions[0]
    chart_height = max(420, 32 * len(top_predictions) + 120)

    main_title = f"<b>Next Token Prediction</b> ({title_suffix})" if title_suffix else f"<b>Next Token Prediction (Top {len(top_predictions)} Candidates)</b>"

    fig.update_layout(
        title=dict(
            text=f"{main_title}<br><sup>Top Candidate: <b>'{top_1_item['token']}'</b> with <b>{top_1_item['probability_pct_str']}</b> probability</sup>",
            x=0.02,
        ),
        xaxis=dict(
            title="<b>Softmax Probability (%)</b> →",
            ticksuffix="%",
            range=[0, x_max_limit],
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)",
            zeroline=True,
            zerolinecolor="rgba(128, 128, 128, 0.4)",
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            tickfont=dict(size=11, family="Courier New, monospace"),
        ),
        hovermode="closest",
        margin=dict(l=180, r=40, t=65, b=45),
        height=chart_height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def plot_generation_flow(
    prompt_tokens: List[Dict[str, Any]],
    generated_tokens: List[Dict[str, Any]],
) -> go.Figure:
    """
    Renders an interactive sequential timeline chart illustrating the autoregressive
    token-by-token generation flow, clearly distinguishing prompt tokens from generated tokens.
    """
    total_tokens = len(prompt_tokens) + len(generated_tokens)
    if total_tokens == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No token generation data available.",
            showarrow=False,
            font=dict(size=14, color="gray"),
        )
        fig.update_layout(height=350)
        return fig

    indices = []
    labels = []
    hover_texts = []
    colors = []

    for p in prompt_tokens:
        idx = p.get("index", len(indices))
        indices.append(idx)
        labels.append(p.get("token_display", ""))
        colors.append("#2c3e50")
        hover_texts.append(
            f"<b>[Prompt Token #{idx}]</b><br>"
            f"<b>Token:</b> {p.get('token_display', '')}<br>"
            f"<b>Token ID:</b> {p.get('token_id', '')}<br>"
            f"<b>Raw Repr:</b> {p.get('token_repr', '')}<br>"
            f"<b>Status:</b> Input Context"
        )

    p_len = len(prompt_tokens)
    for g in generated_tokens:
        step = g.get("step", 1)
        idx = p_len + step - 1
        indices.append(idx)
        labels.append(g.get("token_display", ""))
        colors.append("#00ADB5")
        cum_preview = g.get("cumulative_response", "")[:80]
        hover_texts.append(
            f"<b>[Generated Step #{step}]</b><br>"
            f"<b>Token:</b> {g.get('token_display', '')}<br>"
            f"<b>Token ID:</b> {g.get('token_id', '')}<br>"
            f"<b>Raw Repr:</b> {g.get('token_repr', '')}<br>"
            f"<b>Cumulative Response:</b> {cum_preview}..."
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=indices,
            y=[1] * len(indices),
            mode="lines+markers+text",
            text=[f"{l}" for l in labels],
            textposition="top center",
            textfont=dict(size=10, family="Courier New, monospace"),
            marker=dict(
                size=14,
                color=colors,
                line=dict(color="white", width=1.5),
            ),
            line=dict(color="rgba(150, 150, 150, 0.4)", width=2, dash="dot"),
            hoverinfo="text",
            hovertext=hover_texts,
            showlegend=False,
        )
    )

    if prompt_tokens:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color="#2c3e50", symbol="square"),
                name=f"Prompt Tokens ({len(prompt_tokens)})",
            )
        )
    if generated_tokens:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color="#00ADB5", symbol="square"),
                name=f"Generated Tokens ({len(generated_tokens)})",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"<b>Autoregressive Generation Pipeline ({len(prompt_tokens)} Prompt Tokens → {len(generated_tokens)} Generated Tokens)</b>",
            x=0.02,
        ),
        xaxis=dict(
            title="<b>Sequential Token Position (Time Step t)</b> →",
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)",
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[0.7, 1.3],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
        hovermode="closest",
        margin=dict(l=40, r=40, t=70, b=45),
        height=280,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig
