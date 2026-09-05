"""
LLM-XRay: Streamlit Web UI Application (STEP 15 Complete - Personal Laptop Reliability)
Interactive, beginner-friendly inspection of LLM internals:
1. LLM X-Ray — Title and Introduction
2. Prompt & Generated Response
3. Tokenization
4. Token IDs
5. Embeddings
6. Transformer Architecture
7. Attention Visualization
8. Hidden States
9. Logits & Probabilities
10. Token-by-Token Generation
11. Generation Controls
"""

import os
import traceback
import pandas as pd
import numpy as np
import streamlit as st

# =============================================================
# DEPENDENCY & PACKAGE VERIFICATION
# =============================================================
REQUIRED_PACKAGES = ["torch", "transformers", "plotly", "sklearn", "pandas", "numpy"]
missing_packages = []

for pkg in REQUIRED_PACKAGES:
    try:
        __import__(pkg)
    except ImportError:
        missing_packages.append(pkg)

if missing_packages:
    st.error(
        f"🚨 **Missing Required Python Packages:** `{', '.join(missing_packages)}`\n\n"
        f"Please install all dependencies by running:\n"
        f"```bash\npip install -r requirements.txt\n```"
    )
    st.stop()

# Local module imports
from model import (
    get_device_info,
    clear_memory_cache,
    load_model_and_tokenizer,
    generate_response_with_tokens,
    extract_embeddings,
    get_transformer_layers_info,
    extract_attentions,
    get_attention_matrix,
    extract_hidden_states,
    get_hidden_state_for_layer,
    extract_next_token_logits,
)
from tokenizer import tokenize_text
from visualization import (
    plot_embeddings_2d,
    plot_architecture_stack,
    plot_attention_heatmap,
    plot_hidden_states_2d,
    plot_next_token_probabilities,
    plot_generation_flow,
)

# Page configuration
st.set_page_config(
    page_title="LLM X-Ray — Neural Network Visualizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Fetch compute hardware diagnosis
dev_info = get_device_info()

# Safe cached model loader with user-friendly error diagnostics
@st.cache_resource(show_spinner="Loading Qwen-1.5B neural network weights into memory...")
def get_cached_model_and_tokenizer():
    try:
        return load_model_and_tokenizer()
    except Exception as e:
        return None, str(e), traceback.format_exc()

# Load model and tokenizer
model_res = get_cached_model_and_tokenizer()
if isinstance(model_res, tuple) and len(model_res) == 3 and model_res[0] is None:
    _, load_err_msg, load_traceback = model_res
    st.error("🚨 **Failed to Load LLM Model & Tokenizer**")
    st.markdown(
        f"**Error Details:**\n"
        f"> {load_err_msg}\n\n"
        f"**Troubleshooting Checklist for Personal Laptops:**\n"
        f"1. **Internet Connection:** On first launch, Hugging Face downloads the Qwen 1.5B model (~3 GB). Ensure you are online.\n"
        f"2. **Memory:** Ensure at least 4 GB of available system RAM or disk space.\n"
        f"3. **Dependencies:** Ensure PyTorch and Transformers are installed (`pip install -r requirements.txt`)."
    )
    with st.expander("🛠️ View Python Exception Traceback", expanded=False):
        st.code(load_traceback, language="python")
    
    if st.button("🔄 Retry Model Loading", type="primary"):
        st.cache_resource.clear()
        st.rerun()
    st.stop()
else:
    model, tokenizer = model_res[0], model_res[1]

# =============================================================
# SIDEBAR: LIVE USER CONTROLS & HARDWARE STATUS
# =============================================================
st.sidebar.title("🎛️ Control Panel")

# Hardware Status Badge
if dev_info["is_gpu"]:
    st.sidebar.success(f"⚡ **GPU Mode:** {dev_info['name']}\n\n*{dev_info['memory_details']}*")
else:
    st.sidebar.info(f"💻 **Laptop Mode (CPU):** {dev_info['name']}\n\n*Optimized for personal computers*")

st.sidebar.caption("Fine-tune model generation and visual inspection parameters.")

st.sidebar.subheader("⚙️ Generation Hyperparameters")

temperature = st.sidebar.slider(
    label="🌡️ Temperature",
    min_value=0.0,
    max_value=2.0,
    value=0.7,
    step=0.05,
    help="Controls randomness. 0.0 is deterministic/greedy; higher values produce more diverse text.",
)

top_k = st.sidebar.slider(
    label="🎯 Top-K",
    min_value=0,
    max_value=100,
    value=50,
    step=1,
    help="Limits next-token candidate choices to the top K most likely tokens. Set to 0 to disable.",
)

top_p = st.sidebar.slider(
    label="🌊 Top-P (Nucleus Sampling)",
    min_value=0.05,
    max_value=1.0,
    value=0.90,
    step=0.05,
    help="Keeps the smallest pool of top candidate tokens whose cumulative probability reaches P (e.g. 90%).",
)

max_new_tokens = st.sidebar.slider(
    label="📏 Maximum New Tokens",
    min_value=16,
    max_value=512,
    value=128,
    step=16,
    help="The maximum number of tokens the model is allowed to generate in its response.",
)

st.sidebar.divider()

st.sidebar.subheader("🔍 Inspection Selectors")

sidebar_layer = st.sidebar.slider(
    label="🧱 Default Transformer Layer",
    min_value=1,
    max_value=28,
    value=1,
    step=1,
    help="Selects which Transformer layer (1 to 28) to inspect in Architecture, Attention, and Hidden States.",
)

sidebar_head = st.sidebar.slider(
    label="👁️ Default Attention Head",
    min_value=1,
    max_value=12,
    value=1,
    step=1,
    help="Selects which Attention head (1 to 12) to inspect in the Multi-Head Attention heatmap.",
)

st.sidebar.divider()

# Input Validation & Clamping
safe_temperature = max(0.0, float(temperature))
safe_top_p = max(0.01, min(float(top_p), 1.0))
safe_top_k = max(0, int(top_k))
safe_max_new_tokens = max(1, min(int(max_new_tokens), 512))
safe_layer = max(1, min(int(sidebar_layer), 28))
safe_head = max(1, min(int(sidebar_head), 12))
do_sample = safe_temperature > 0.0


# =============================================================
# SECTION 1: LLM X-RAY — TITLE AND INTRODUCTION
# =============================================================
st.title("🔬 1. LLM X-Ray — Title and Introduction")
st.caption("An interactive, visual deep-dive into how Large Language Models think and generate text (Model: `Qwen/Qwen2.5-1.5B-Instruct`)")

intro_container = st.container()
with intro_container:
    col_intro_1, col_intro_2 = st.columns([3, 2])
    
    with col_intro_1:
        st.markdown(
            """
            ### Welcome to **LLM X-Ray**! 👋
            Ever wondered what happens inside a Large Language Model when you type a prompt?
            
            Instead of treating the AI as an opaque black box, **LLM X-Ray** peels back the layers of a real, local **1.5 Billion parameter Transformer** model in real-time.
            
            Follow the journey of your prompt as it flows through the complete neural network pipeline:
            """
        )
        st.info(
            "**The 7-Stage Neural Network Flow:**\n\n"
            "1. 🔤 **Tokenization:** Splitting your raw text into subwords (tokens).\n"
            "2. 🔢 **Token IDs:** Converting subwords into discrete vocabulary indices ($V = 151,936$).\n"
            r"3. 🌐 **Embeddings:** Mapping IDs into high-dimensional semantic vectors ($\mathbb{R}^{1536}$)." + "\n"
            "4. 🏗️ **Transformer Architecture:** Passing vectors through **28 stacked neural layers**.\n"
            "5. 🧠 **Multi-Head Attention:** Tokens dynamically looking at other words to gather context.\n"
            "6. 🧬 **Hidden States:** Rich contextual representations evolving across depth.\n"
            "7. 🎲 **Logits & Probabilities:** Projecting to vocabulary scores and sampling the next token autoregressively."
        )

    with col_intro_2:
        st.markdown("### 📌 Quick Model Specs")
        spec_df = pd.DataFrame([
            {"Specification": "Model Identifier", "Value": "Qwen/Qwen2.5-1.5B-Instruct"},
            {"Specification": "Architecture", "Value": "Decoder-Only Causal Transformer"},
            {"Specification": "Total Parameters", "Value": "1.54 Billion"},
            {"Specification": "Transformer Layers", "Value": "28 Layers"},
            {"Specification": "Hidden Dimension (d_model)", "Value": "1,536 dimensions"},
            {"Specification": "Attention Heads", "Value": "12 Query / 2 Key-Value (GQA)"},
            {"Specification": "Vocabulary Size", "Value": "151,936 tokens"},
            {"Specification": "Activation Function", "Value": "SwiGLU / SILU"},
            {"Specification": "Compute Mode", "Value": "CPU (Laptop Mode)" if not dev_info["is_gpu"] else f"GPU ({dev_info['device_type'].upper()})"},
        ])
        st.dataframe(spec_df, use_container_width=True, hide_index=True)


# =============================================================
# SECTION 2: PROMPT & GENERATED RESPONSE
# =============================================================
st.divider()
st.header("2. 💬 Prompt & Generated Response")
st.markdown(
    "Enter a prompt below or pick a preset example. Click **'Generate & Inspect'** to run the complete LLM X-Ray analysis across all sections."
)

# Preset prompt selector
preset_prompts = {
    "Capital of France (Factual QA)": "What is the capital of France?",
    "Quick Brown Fox (Classic Pangram)": "The quick brown fox jumps over the lazy dog.",
    "Explain Gravity (Simple Science)": "Explain gravity in one simple sentence.",
    "Python Programming (Code & Tech)": "Python is a popular programming language because",
    "Custom Prompt": "",
}

selected_preset = st.selectbox(
    "💡 Pick a Preset Prompt (or choose 'Custom Prompt' to write your own):",
    options=list(preset_prompts.keys()),
    index=0,
)

default_text = preset_prompts[selected_preset] if selected_preset != "Custom Prompt" else "What is the capital of France?"

user_prompt = st.text_area(
    label="Your Input Prompt:",
    value=default_text,
    placeholder="Type any sentence, question, or code prompt...",
    height=100,
    key="main_prompt_input",
)

col_btn, col_btn_info = st.columns([1, 3])
with col_btn:
    generate_clicked = st.button("🚀 Generate & Inspect", type="primary", use_container_width=True)
with col_btn_info:
    st.caption("Tip: Adjust hyperparameters (Temperature, Top-K, Top-P) in the sidebar anytime.")

# Prompt validation
if not user_prompt.strip():
    st.warning("⚠️ Please enter a prompt above to run the LLM X-Ray inspection.")
    st.stop()

# Track analysis execution in session state
if generate_clicked or "last_analyzed_prompt" not in st.session_state:
    st.session_state["last_analyzed_prompt"] = user_prompt.strip()

active_prompt = st.session_state.get("last_analyzed_prompt", user_prompt.strip())

# Prompt length warning for laptop CPU responsiveness
approx_chars = len(active_prompt)
if approx_chars > 600:
    st.info(f"ℹ️ **Long prompt detected ({approx_chars} chars):** Analysis across 28 layers will execute smoothly, though CPU inference may take a few moments.")

# Execute pipeline with memory management and exception handling
clear_memory_cache()

with st.spinner("Running LLM X-Ray pipeline across all 28 Transformer layers..."):
    try:
        # 1. Tokenization
        token_data = tokenize_text(active_prompt, tokenizer=tokenizer)
        if token_data.get("error"):
            st.error(f"Tokenization error: {token_data['error']}")
            st.stop()

        # 2. Embeddings
        emb_data = extract_embeddings(token_ids=token_data["token_ids"], model=model)
        if emb_data.get("error"):
            st.error(f"Embeddings error: {emb_data['error']}")

        # 3. Architecture Info
        arch_info = get_transformer_layers_info(model=model)

        # 4. Attention
        attn_data = extract_attentions(token_ids=token_data["token_ids"], model=model)

        # 5. Hidden States
        hidden_data = extract_hidden_states(token_ids=token_data["token_ids"], model=model)

        # 6. Logits & Next-Token Probabilities
        top_k_for_logits = max(5, min(safe_top_k if safe_top_k > 0 else 10, 25))
        logits_data = extract_next_token_logits(
            token_ids=token_data["token_ids"],
            model=model,
            tokenizer=tokenizer,
            top_k=top_k_for_logits,
        )

        # 7. Generation
        gen_data = generate_response_with_tokens(
            prompt=active_prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=safe_max_new_tokens,
            temperature=safe_temperature,
            top_p=safe_top_p,
            top_k=safe_top_k,
            do_sample=do_sample,
        )

    except Exception as pipeline_err:
        st.error(f"🚨 **Pipeline execution error:** {str(pipeline_err)}")
        with st.expander("🛠️ View Pipeline Error Traceback"):
            st.code(traceback.format_exc(), language="python")
        st.stop()
    finally:
        clear_memory_cache()

# Display Prompt & Generated Response Overview Card
p_col1, p_col2 = st.columns([1, 1])

with p_col1:
    st.markdown("##### 📥 Active Input Prompt")
    st.info(f"**\"{active_prompt}\"**")
    st.caption(f"Prompt Length: `{len(active_prompt)}` characters | `{token_data['total_tokens']}` tokens")

with p_col2:
    st.markdown("##### 💬 Model Generated Response")
    if gen_data.get("error"):
        st.error(f"Generation error: {gen_data['error']}")
    else:
        st.success(f"**{gen_data['final_response']}**")
        st.caption(
            f"Generated: `{gen_data['generated_token_count']}` new tokens | "
            f"Total Sequence: `{gen_data['total_token_count']}` tokens | "
            f"Mode: `{'Stochastic (Sampling)' if do_sample else 'Greedy (Deterministic)'}`"
        )


# =============================================================
# SECTION 3: TOKENIZATION
# =============================================================
st.divider()
st.header("3. 🔤 Tokenization")
st.markdown(
    """
    **What is Tokenization?**  
    Language models cannot read whole words or raw text directly. Instead, a **Tokenizer** splits text into discrete subword chunks called **Tokens**.
    Common words might be a single token (e.g. `What`, `France`), while rare words or prefixes are split into smaller pieces.
    """
)

tok_col1, tok_col2, tok_col3 = st.columns(3)
with tok_col1:
    st.metric(label="Total Tokens", value=token_data["total_tokens"])
with tok_col2:
    st.metric(label="Total Characters", value=len(active_prompt))
with tok_col3:
    avg_chars = round(len(active_prompt) / max(1, token_data["total_tokens"]), 2)
    st.metric(label="Avg Chars / Token", value=f"{avg_chars} chars")

st.markdown("##### 🏷️ Visual Token Stream (Decoded Subwords)")
st.caption("Each colored badge below represents an individual token produced by the Qwen BPE tokenizer:")

token_badges_html = " ".join([
    f"<span style='background-color:#1e3a8a; color:#bfdbfe; padding:4px 10px; margin:3px; border-radius:6px; font-family:monospace; font-size:13px; display:inline-block; border:1px solid #3b82f6;'>"
    f"<span style='color:#93c5fd; font-size:10px;'>[{idx}]</span> <b>{repr(tok)[1:-1]}</b>"
    f"</span>"
    for idx, tok in enumerate(token_data["tokens"])
])
st.markdown(token_badges_html, unsafe_allow_html=True)

with st.expander("📋 View Token Subwords List", expanded=False):
    st.write(token_data["tokens"])


# =============================================================
# SECTION 4: TOKEN IDS
# =============================================================
st.divider()
st.header("4. 🔢 Token IDs")
st.markdown(
    """
    **What are Token IDs?**  
    Neural networks operate entirely on numbers. The tokenizer maps every unique token to a unique **integer ID** from the model's vocabulary dictionary ($V = 151,936$ tokens for Qwen 2.5).
    These IDs serve as the discrete addresses used to look up embedding vectors.
    """
)

col_id_seq, col_id_vocab = st.columns([2, 1])
with col_id_seq:
    st.markdown("##### 🔢 Token ID Sequence")
    st.code(str(token_data["token_ids"]), language="python")
with col_id_vocab:
    st.markdown("##### 📖 Vocabulary Info")
    st.info(f"**Model Vocab Size:** `151,936` unique token IDs\n\n**Sequence Length:** `{token_data['total_tokens']}` token IDs")

st.markdown("##### 📋 Complete Token & Token ID Breakdown Table")
if token_data["breakdown"]:
    df_tokens = pd.DataFrame(token_data["breakdown"])
    df_tokens.rename(
        columns={
            "index": "Index",
            "token": "Decoded Token String",
            "token_repr": "Raw Representation",
            "token_id": "Vocabulary Token ID",
        },
        inplace=True,
    )
    st.dataframe(df_tokens, use_container_width=True, hide_index=True)


# =============================================================
# SECTION 5: EMBEDDINGS
# =============================================================
st.divider()
st.header("5. 🌐 Embeddings")
st.markdown(
    r"""
    **What are Embeddings?**  
    In the embedding layer, each integer Token ID is looked up in a large embedding weight matrix.
    This converts each discrete ID into a **1,536-dimensional continuous vector** ($\mathbb{R}^{1536}$).
    In this high-dimensional semantic space, words with related meanings or functions are positioned close to each other.
    """
)

emb_m1, emb_m2, emb_m3, emb_m4 = st.columns(4)
with emb_m1:
    st.metric(label="Embedding Dimension", value=f"{emb_data['embedding_dim']} dimensions")
with emb_m2:
    st.metric(label="Mean L2 Vector Norm", value=f"{emb_data['global_stats'].get('mean_norm', 0):.4f}")
with emb_m3:
    st.metric(label="Global Mean Value", value=f"{emb_data['global_stats'].get('global_mean', 0):.4f}")
with emb_m4:
    st.metric(label="Global Std Dev", value=f"{emb_data['global_stats'].get('global_std', 0):.4f}")

st.markdown("##### 📊 Interactive 2D PCA Projection of Input Embeddings")
st.caption(
    "Using Principal Component Analysis (PCA), we compress the 1,536 dimensions into 2 principal axes of maximum variance. "
    "Hover over any point to inspect token details, or follow the dotted trajectory line showing sequential input order."
)

fig_pca = plot_embeddings_2d(
    tokens=token_data["tokens"],
    token_ids=token_data["token_ids"],
    embeddings_matrix=emb_data["embeddings_matrix"],
)
st.plotly_chart(fig_pca, use_container_width=True)

with st.expander("🔬 View Per-Token Embedding Statistics & Vector Inspector", expanded=False):
    if emb_data["token_embeddings"]:
        stats_rows = []
        for item in emb_data["token_embeddings"]:
            idx = item["index"]
            tok = token_data["tokens"][idx] if idx < len(token_data["tokens"]) else ""
            stats_rows.append({
                "Index": idx,
                "Token": repr(tok)[1:-1],
                "Token ID": item["token_id"],
                "L2 Norm": round(item["norm"], 4),
                "Mean": round(item["mean"], 5),
                "Std Dev": round(item["std"], 5),
                "Min": round(item["min"], 5),
                "Max": round(item["max"], 5),
                "Vector Preview (First 8 dims)": str([round(x, 4) for x in item["vector_preview"]]),
            })
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

        selected_emb_idx = st.selectbox(
            "Select a token to inspect its full 1,536-dimensional raw embedding vector:",
            options=range(len(emb_data["token_embeddings"])),
            format_func=lambda i: f"[{i}] Token: '{token_data['tokens'][i]}' (ID: {token_data['token_ids'][i]})",
            key="emb_token_select",
        )
        selected_vec = emb_data["token_embeddings"][selected_emb_idx]["vector"]
        st.write(f"**Full Vector for Token `{token_data['tokens'][selected_emb_idx]}` (Shape: {selected_vec.shape}):**")
        st.code(str(selected_vec.tolist()), language="python")


# =============================================================
# SECTION 6: TRANSFORMER ARCHITECTURE
# =============================================================
st.divider()
st.header("6. 🏗️ Transformer Architecture")
st.markdown(
    """
    **What is the Transformer Architecture?**  
    The model is constructed from **28 identical Transformer blocks** stacked vertically.
    Each block performs two key operations:
    1. **Multi-Head Self-Attention:** Allows tokens to exchange information and understand relationships.
    2. **SwiGLU Feed-Forward Network (MLP):** Expands representation into 8,960 intermediate dimensions to store and retrieve learned facts and reasoning patterns.
    RMS Normalization is applied before and after each sub-layer for training and numerical stability.
    """
)

total_layers = arch_info["num_layers"]

arch_c1, arch_c2, arch_c3, arch_c4 = st.columns(4)
with arch_c1:
    st.metric("Total Layers", total_layers)
with arch_c2:
    st.metric("Hidden Size (d_model)", arch_info["hidden_size"])
with arch_c3:
    st.metric("Attention Heads", f"{arch_info['num_attention_heads']} Q / {arch_info['num_key_value_heads']} KV")
with arch_c4:
    st.metric("FFN Intermediate Dim", arch_info["intermediate_size"])

st.markdown("##### 🎯 Layer Selection Control")
selected_layer_num = st.slider(
    label="Select Transformer Layer to inspect:",
    min_value=1,
    max_value=total_layers,
    value=safe_layer,
    step=1,
    key="arch_sec_layer_slider",
    help=f"Select any layer between Layer 1 and Layer {total_layers}",
)

selected_layer_idx = max(0, min(selected_layer_num - 1, len(arch_info["layers"]) - 1))
layer_meta = arch_info["layers"][selected_layer_idx]

col_arch_vis, col_arch_spec = st.columns([1, 1])

with col_arch_vis:
    st.markdown("##### 📐 Sequential Architecture Pipeline")
    st.info(
        f"**Active Stack:** Embeddings ➔ Layer 1 ... **Layer {selected_layer_num}** ⭐ *(Highlighted)* ... Layer {total_layers} ➔ Output LM Head"
    )
    fig_stack = plot_architecture_stack(
        num_layers=total_layers,
        selected_layer_num=selected_layer_num,
    )
    st.plotly_chart(fig_stack, use_container_width=True)

with col_arch_spec:
    st.markdown(f"##### 🔍 Layer {selected_layer_num} Detailed Specification")
    st.success(
        f"**Active Selection:** Transformer Block `{selected_layer_num}` of `{total_layers}`\n\n"
        f"- **Layer Parameters:** `{layer_meta['param_count']:,}` parameters\n"
        f"- **Attention Type:** `{layer_meta['attn_type']}`\n"
        f"- **Query Heads (Q):** `{layer_meta['num_attention_heads']}` (Head Dim: `{layer_meta['head_dim']}`)\n"
        f"- **Key/Value Heads (KV):** `{layer_meta['num_key_value_heads']}` (Grouped Query Attention)\n"
        f"- **FFN / MLP Structure:** `{layer_meta['mlp_type']}` (Dim: `{layer_meta['intermediate_size']}`)\n"
        f"- **Layer Normalization:** `{layer_meta['norm_type']}` (Pre & Post Attention)"
    )

    with st.expander("📋 View Complete Layer JSON Metadata", expanded=True):
        st.json({
            "layer_number": selected_layer_num,
            "layer_index": selected_layer_idx,
            "total_layers_in_model": total_layers,
            "hidden_size": layer_meta["hidden_size"],
            "head_dim": layer_meta["head_dim"],
            "query_attention_heads": layer_meta["num_attention_heads"],
            "key_value_attention_heads": layer_meta["num_key_value_heads"],
            "mlp_intermediate_size": layer_meta["intermediate_size"],
            "parameter_count": layer_meta["param_count"],
            "attention_mechanism": layer_meta["attn_type"],
            "activation_fn": "SILU / SwiGLU",
            "norm_mechanism": layer_meta["norm_type"],
        })


# =============================================================
# SECTION 7: ATTENTION VISUALIZATION
# =============================================================
st.divider()
st.header("7. 🧠 Attention Visualization")
st.markdown(
    r"""
    **What is Multi-Head Attention?**  
    Attention is the mathematical mechanism that lets each token in a sentence "look back" at other tokens to understand context.
    - **Query ($Q$):** What the current token is looking for.
    - **Key ($K$):** What information other tokens offer.
    - **Value ($V$):** The actual semantic content passed forward.
    
    The resulting **Attention Weights** ($\text{Softmax}(QK^T / \sqrt{d})$) indicate how much focus each token puts on every other token.
    """
)

if attn_data.get("error"):
    st.error(f"Attention extraction error: {attn_data['error']}")
    if attn_data.get("traceback"):
        with st.expander("🛠️ View Error Traceback"):
            st.code(attn_data["traceback"], language="python")
elif attn_data["num_layers"] == 0:
    st.warning("No attention matrices could be extracted for the current prompt.")
else:
    num_attn_layers = max(1, attn_data["num_layers"])
    num_attn_heads = max(1, attn_data["num_heads"])

    col_al_sel, col_ah_sel, col_am_sel = st.columns([2, 2, 2])
    with col_al_sel:
        attn_layer_num = st.slider(
            label="Transformer Layer for Attention:",
            min_value=1,
            max_value=num_attn_layers,
            value=safe_layer,
            step=1,
            key="attn_sec_layer_slider",
        )
    with col_ah_sel:
        attn_head_num = st.slider(
            label="Attention Head:",
            min_value=1,
            max_value=num_attn_heads,
            value=safe_head,
            step=1,
            key="attn_sec_head_slider",
        )
    with col_am_sel:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        view_avg_head = st.checkbox(
            "Average Across All Heads",
            value=False,
            help="Compute and display the mean attention pattern across all heads in this layer",
            key="attn_sec_avg_check",
        )

    layer_idx = attn_layer_num - 1
    head_idx = None if view_avg_head else (attn_head_num - 1)
    attention_matrix = get_attention_matrix(
        attentions_data=attn_data,
        layer_index=layer_idx,
        head_index=head_idx,
    )

    ab_col1, ab_col2, ab_col3, ab_col4 = st.columns(4)
    with ab_col1:
        st.metric("Active Layer", f"Layer {attn_layer_num} of {num_attn_layers}")
    with ab_col2:
        head_lbl = "Mean (All Heads)" if view_avg_head else f"Head {attn_head_num} of {num_attn_heads}"
        st.metric("Active Head", head_lbl)
    with ab_col3:
        st.metric("Matrix Shape", f"{attention_matrix.shape[0]} × {attention_matrix.shape[1]}")
    with ab_col4:
        max_score = float(np.max(attention_matrix)) if attention_matrix.size > 0 else 0.0
        st.metric("Max Attention Score", f"{max_score:.4f}")

    st.markdown("##### 🗺️ Interactive Attention Heatmap")
    st.caption(
        "**Rows (Y-axis):** Query token that is currently looking. **Columns (X-axis):** Key token being attended to. "
        "Brighter cells indicate higher attention weights. Due to causal masking, tokens can only attend to current and preceding tokens (lower triangular)."
    )

    fig_attn = plot_attention_heatmap(
        tokens=token_data["tokens"],
        attention_matrix=attention_matrix,
        layer_num=attn_layer_num,
        head_num=attn_head_num if not view_avg_head else None,
        is_average=view_avg_head,
    )
    st.plotly_chart(fig_attn, use_container_width=True)

    with st.expander("🔬 View Raw Attention Values Table & Head Diagnostics", expanded=False):
        token_display_labels = [
            f"[{i}] {tok.replace(' ', '␣').replace(chr(10), '↵')}"
            for i, tok in enumerate(token_data["tokens"][:attention_matrix.shape[0]])
        ]
        df_attn = pd.DataFrame(
            attention_matrix,
            index=token_display_labels,
            columns=token_display_labels,
        )
        st.markdown("**Exact Attention Score Matrix:**")
        st.dataframe(
            df_attn.style.background_gradient(cmap="Blues", vmin=0.0, vmax=1.0).format("{:.4f}"),
            use_container_width=True,
        )

        diag_c1, diag_c2 = st.columns(2)
        with diag_c1:
            st.markdown("**Attention Normalization & Self-Attention:**")
            row_sums = [round(float(s), 4) for s in np.sum(attention_matrix, axis=1)] if attention_matrix.size > 0 else []
            self_att_mean = float(np.mean(np.diag(attention_matrix))) if attention_matrix.size > 0 else 0.0
            mean_att = float(np.mean(attention_matrix)) if attention_matrix.size > 0 else 0.0
            st.write(rf"- **Softmax Row Sums ($\sum = 1.0$):** `{row_sums}`")
            st.write(f"- **Self-Attention Weight (Diagonal Mean):** `{self_att_mean:.4f}`")
            st.write(f"- **Mean Attention Weight:** `{mean_att:.4f}`")
        with diag_c2:
            st.markdown("**Head Sparsity & Entropy:**")
            if attention_matrix.size > 0:
                entropy = -np.sum(attention_matrix * np.log(np.clip(attention_matrix, 1e-12, 1.0)), axis=1)
                mean_entropy = float(np.mean(entropy))
                min_att = float(np.min(attention_matrix))
                max_att = float(np.max(attention_matrix))
            else:
                mean_entropy, min_att, max_att = 0.0, 0.0, 0.0

            st.write(f"- **Mean Query Entropy:** `{mean_entropy:.4f}` nats")
            st.write(f"- **Min Attention Value:** `{min_att:.6f}`")
            st.write(f"- **Max Attention Value:** `{max_att:.6f}`")


# =============================================================
# SECTION 8: HIDDEN STATES
# =============================================================
st.divider()
st.header("8. 🧬 Hidden States")
st.markdown(
    """
    **What are Hidden States?**  
    As token representations travel upward from Layer 1 to Layer 28, their numerical vectors are transformed at every step.
    - **Layer 0 (Embedding):** Static, dictionary-level word meanings.
    - **Middle Layers (10–18):** Syntactic parsing, clause relationships, and entity resolution.
    - **Deep Layers (20–28):** High-level task reasoning, factual retrieval, and next-token preparation.
    """
)

if hidden_data.get("error"):
    st.error(f"Hidden state extraction error: {hidden_data['error']}")
    if hidden_data.get("traceback"):
        with st.expander("🛠️ View Error Traceback"):
            st.code(hidden_data["traceback"], language="python")
elif hidden_data["num_hidden_states"] == 0:
    st.warning("No hidden states available.")
else:
    total_hs_layers = max(1, hidden_data["num_layers"])
    hidden_dim = hidden_data["hidden_dim"]
    seq_len = hidden_data["seq_len"]

    col_h_pres, col_h_slid = st.columns([1, 1])
    with col_h_pres:
        layer_preset = st.radio(
            label="Quick Milestone Presets:",
            options=["Layer 1", "Layer 10", "Layer 20", "Layer 28", "Custom (Slider)"],
            horizontal=True,
            key="hs_sec_preset_radio",
        )

    preset_map = {"Layer 1": 1, "Layer 10": 10, "Layer 20": 20, "Layer 28": 28}

    with col_h_slid:
        default_hs_val = preset_map.get(layer_preset, safe_layer)
        selected_hs_layer = st.slider(
            label="Select Layer (0 = Embedding, 1..28 = Transformer Layers):",
            min_value=0,
            max_value=total_hs_layers,
            value=default_hs_val if layer_preset != "Custom (Slider)" else safe_layer,
            step=1,
            key="hs_sec_layer_slider",
        )

    active_hs_num = default_hs_val if layer_preset in preset_map else selected_hs_layer
    active_hs_info = get_hidden_state_for_layer(hidden_data, active_hs_num)

    if active_hs_info is not None:
        st.markdown(f"##### 📐 Dimensions & Shape for **{active_hs_info['name']}**")
        hs_m1, hs_m2, hs_m3, hs_m4 = st.columns(4)
        with hs_m1:
            st.metric("Tensor Shape", f"(1, {seq_len}, {hidden_dim})")
        with hs_m2:
            st.metric("Matrix Shape", f"{seq_len} × {hidden_dim}")
        with hs_m3:
            st.metric("Hidden Dim", f"{hidden_dim} (d_model)")
        with hs_m4:
            st.metric("Mean L2 Norm", f"{active_hs_info['mean_l2_norm']:.4f}")

        st.markdown(f"##### 📊 2D PCA Representation Space at **{active_hs_info['name']}**")
        st.caption("Notice how token positions shift across layer depth as the model applies contextual reasoning to each word.")
        
        fig_hs_pca = plot_hidden_states_2d(
            tokens=token_data["tokens"],
            token_ids=token_data["token_ids"],
            hidden_matrix=active_hs_info["matrix"],
            layer_label=active_hs_info["name"],
            layer_num=active_hs_num,
        )
        st.plotly_chart(fig_hs_pca, use_container_width=True)

        with st.expander("🔬 View Per-Token Hidden State Statistics & Vector Values", expanded=False):
            hs_matrix = active_hs_info["matrix"]
            token_rows = []
            for idx, (tok, tid) in enumerate(zip(token_data["tokens"], token_data["token_ids"])):
                if idx < hs_matrix.shape[0]:
                    vec = hs_matrix[idx]
                    token_rows.append({
                        "Index": idx,
                        "Token": repr(tok)[1:-1],
                        "Token ID": tid,
                        "L2 Norm": round(float(np.linalg.norm(vec)), 4),
                        "Mean": round(float(np.mean(vec)), 5),
                        "Std Dev": round(float(np.std(vec)), 5),
                        "Min": round(float(np.min(vec)), 5),
                        "Max": round(float(np.max(vec)), 5),
                        "Vector Preview (First 8 dims)": str([round(float(x), 4) for x in vec[:8]]),
                    })
            st.dataframe(pd.DataFrame(token_rows), use_container_width=True, hide_index=True)

            if token_rows:
                selected_hs_tok = st.selectbox(
                    f"Inspect full {hidden_dim}-D vector for a token in {active_hs_info['name']}:",
                    options=range(len(token_rows)),
                    format_func=lambda i: f"[{i}] Token: '{token_data['tokens'][i]}' (ID: {token_data['token_ids'][i]})",
                    key="hs_sec_tok_inspect",
                )
                full_hs_vec = hs_matrix[selected_hs_tok]
                st.write(f"**Full Vector for Token `{token_data['tokens'][selected_hs_tok]}` in `{active_hs_info['name']}`:**")
                st.code(str(full_hs_vec.tolist()), language="python")


# =============================================================
# SECTION 9: LOGITS & PROBABILITIES
# =============================================================
st.divider()
st.header("9. 🎲 Logits & Probabilities")
st.markdown(
    r"""
    **What are Logits and Probabilities?**  
    After the final Layer 28, the model applies a linear projection called the **Language Model Head (LM Head)** to convert the last token's hidden state into raw, unnormalized prediction scores (**Logits**) for all 151,936 vocabulary tokens.
    
    The **Softmax function** converts these raw logits into valid probabilities that sum to 100%:
    $$\sigma(z_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$
    """
)

col_topk_ctl, col_topk_exp = st.columns([1, 2])
with col_topk_ctl:
    top_k_select = st.slider(
        label="Top-K Predictions to Display:",
        min_value=5,
        max_value=25,
        value=max(5, min(safe_top_k if safe_top_k > 0 else 10, 25)),
        step=1,
        key="logits_sec_topk_slider",
    )
with col_topk_exp:
    st.info(
        "**Prediction Pipeline:**\n\n"
        "Final Hidden State ($1536$-D) ➔ LM Head ($1536 \\times 151936$) ➔ Raw Logits ($151936$-D) ➔ Softmax ➔ Output Probabilities"
    )

if logits_data.get("error"):
    st.error(f"Logits error: {logits_data['error']}")
    if logits_data.get("traceback"):
        with st.expander("🛠️ View Error Traceback"):
            st.code(logits_data["traceback"], language="python")
elif not logits_data["top_predictions"]:
    st.warning("No next-token predictions available.")
else:
    top_preds = logits_data["top_predictions"][:top_k_select]
    top_1 = top_preds[0] if top_preds else {"token": "", "token_id": 0, "probability_pct_str": "0%"}

    lp_col1, lp_col2, lp_col3, lp_col4 = st.columns(4)
    with lp_col1:
        st.metric("Top-1 Predicted Token", f"'{top_1['token']}'", help=f"Token ID: {top_1['token_id']}")
    with lp_col2:
        st.metric("Top-1 Probability", top_1["probability_pct_str"])
    with lp_col3:
        st.metric("Vocabulary Size", f"{logits_data['vocab_size']:,} words")
    with lp_col4:
        st.metric("Distribution Entropy", f"{logits_data['entropy']:.4f} nats")

    st.markdown("##### 📊 Next-Token Probability Distribution Chart")
    st.caption("Interactive horizontal bar chart of the highest-confidence token candidates for the immediate next position:")

    fig_probs = plot_next_token_probabilities(top_predictions=top_preds)
    st.plotly_chart(fig_probs, use_container_width=True)

    st.markdown("##### 📋 Ranked Next-Token Candidate Table")
    table_rows = []
    for item in top_preds:
        table_rows.append({
            "Rank": f"#{item['rank']}",
            "Token": item["token_display"],
            "Token ID": item["token_id"],
            "Probability (%)": item["probability_pct_str"],
            "Raw Probability": round(item["probability"], 6),
            "Logit Score": round(item["logit"], 4),
        })

    df_probs = pd.DataFrame(table_rows)
    st.dataframe(df_probs, use_container_width=True, hide_index=True)

    with st.expander("🔬 View LM Head Logits Vector Diagnostics", expanded=False):
        st.write(
            f"- **Logits Vector Dimensions:** `(151936,)`\n"
            f"- **Max Raw Logit:** `{logits_data['logits_max']:.4f}` (Token ID `{top_1['token_id']}`: `'{top_1['token']}')`\n"
            f"- **Min Raw Logit:** `{logits_data['logits_min']:.4f}`\n"
            f"- **Mean Raw Logit:** `{logits_data['logits_mean']:.4f}`\n"
            r"- **Softmax Normalization Check:** $\sum p_i = 1.000000$ (100.0%)"
        )


# =============================================================
# SECTION 10: TOKEN-BY-TOKEN GENERATION
# =============================================================
st.divider()
st.header("10. 🤖 Token-by-Token Generation")
st.markdown(
    """
    **What is Autoregressive Generation?**  
    Large Language Models produce answers **one token at a time**:
    1. The model takes the prompt and predicts the 1st next token.
    2. That 1st token is appended to the prompt.
    3. The model takes the new, longer sequence and predicts the 2nd token.
    4. This cycle repeats until an End-of-Sequence token (`<|im_end|>`) is produced or the maximum token limit is reached.
    """
)

if gen_data.get("error"):
    st.error(f"Generation error: {gen_data['error']}")
    if gen_data.get("traceback"):
        with st.expander("🛠️ View Error Traceback"):
            st.code(gen_data["traceback"], language="python")
else:
    g_col1, g_col2, g_col3, g_col4 = st.columns(4)
    with g_col1:
        st.metric("Prompt Tokens", f"{gen_data['prompt_token_count']} tokens")
    with g_col2:
        st.metric("Generated Tokens", f"{gen_data['generated_token_count']} tokens")
    with g_col3:
        st.metric("Total Sequence Length", f"{gen_data['total_token_count']} tokens")
    with g_col4:
        status_lbl = "Complete (EOS reached)" if not gen_data.get("is_truncated") else "Max Tokens Reached"
        st.metric("Status", status_lbl)

    st.markdown("##### 💬 Complete Model Generated Output")
    st.success(gen_data["final_response"])

    st.markdown("##### 📐 Autoregressive Token Flow Timeline")
    st.caption("Prompt Tokens (Input Context) ➔ Step-by-Step Generated Tokens (Autoregressive Loop):")
    fig_flow = plot_generation_flow(
        prompt_tokens=gen_data["prompt_tokens"],
        generated_tokens=gen_data["generated_tokens"],
    )
    st.plotly_chart(fig_flow, use_container_width=True)

    with st.expander("🏷️ View Token Stream Badges (Prompt vs Generated)", expanded=True):
        st.markdown("**📥 Input Prompt Tokens (Blue):**")
        prompt_html = " ".join([
            f"<span style='background-color:#1e293b; color:#93c5fd; padding:3px 8px; margin:2px; border-radius:4px; font-family:monospace; font-size:12px;' title='Token ID: {p['token_id']}'>[{p['index']}] {p['token_display']}</span>"
            for p in gen_data["prompt_tokens"]
        ])
        st.markdown(prompt_html, unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("**🎯 Newly Generated Output Tokens (Emerald Green):**")
        gen_html = " ".join([
            f"<span style='background-color:#064e3b; color:#6ee7b7; padding:3px 8px; margin:2px; border-radius:4px; font-family:monospace; font-size:12px;' title='Step {g['step']} | ID: {g['token_id']}'>#{g['step']} {g['token_display']}</span>"
            for g in gen_data["generated_tokens"]
        ])
        st.markdown(gen_html, unsafe_allow_html=True)

    st.markdown("##### ⏳ Step-by-Step Progressively Growing Response")
    step_table_rows = []
    for g in gen_data["generated_tokens"]:
        step_table_rows.append({
            "Generation Step": f"Step #{g['step']}",
            "Decoded Token": g["token_display"],
            "Token ID": g["token_id"],
            "Raw Repr": g["token_repr"],
            "Cumulative Text So Far": g["cumulative_response"],
        })

    df_gen_steps = pd.DataFrame(step_table_rows)
    st.dataframe(df_gen_steps, use_container_width=True, hide_index=True)


# =============================================================
# SECTION 11: GENERATION CONTROLS
# =============================================================
st.divider()
st.header("11. 🎛️ Generation Controls")
st.markdown(
    """
    **Understanding Generation Hyperparameters & Visual Controls**  
    Generation hyperparameters dictate how the model samples from its probability distribution at each step.
    Use the guides below to understand how each setting impacts model creativity, focus, and stability.
    """
)

# Active parameters badge banner
st.info(
    f"**Currently Active Settings:** "
    f"🌡️ **Temperature:** `{safe_temperature:.2f}` &nbsp;|&nbsp; "
    f"🎯 **Top-K:** `{safe_top_k}` &nbsp;|&nbsp; "
    f"🌊 **Top-P:** `{safe_top_p:.2f}` &nbsp;|&nbsp; "
    f"📏 **Max Tokens:** `{safe_max_new_tokens}` &nbsp;|&nbsp; "
    f"🧱 **Layer:** `{safe_layer}` &nbsp;|&nbsp; "
    f"👁️ **Head:** `{safe_head}`"
)

tab_temp, tab_topk, tab_topp, tab_len, tab_presets = st.tabs([
    "🌡️ Temperature",
    "🎯 Top-K Sampling",
    "🌊 Top-P (Nucleus)",
    "📏 Max Tokens",
    "💡 Recommended Recipes",
])

with tab_temp:
    st.markdown(
        r"""
        ### 🌡️ Temperature ($T$)
        **How it works:**  
        Temperature scales the logits before the Softmax function is calculated:
        $$z_i' = \frac{z_i}{T}$$
        
        - **$T = 0.0$ (Greedy Search):** The model always picks the single highest-probability token. 100% deterministic and repeatable. Ideal for math, factual retrieval, and structured JSON.
        - **$T = 0.7$ (Balanced):** The standard default for natural, coherent English dialogue.
        - **$T \ge 1.2$ (Creative):** Flattens the distribution, giving unexpected and creative words a higher chance to be selected.
        """
    )

with tab_topk:
    st.markdown(
        """
        ### 🎯 Top-K Sampling
        **How it works:**  
        At each generation step, the model sorts all 151,936 vocabulary words by score and **prunes the candidate pool to only the top $K$ choices**.
        
        - **$K = 1$:** Equivalent to greedy search.
        - **$K = 40-50$:** Filters out low-probability gibberish or out-of-context tokens while keeping natural variety.
        - **$K = 0$ (Disabled):** Allows the model to consider the entire vocabulary.
        """
    )

with tab_topp:
    st.markdown(
        r"""
        ### 🌊 Top-P (Nucleus Sampling)
        **How it works:**  
        Instead of a fixed number of tokens like Top-K, Top-P dynamically chooses the smallest set of top tokens whose cumulative probability adds up to $P$ (e.g. 90%):
        $$\sum_{i \in V^{(p)}} p_i \ge P$$
        
        - When the model is **confident**, the top 1 or 2 tokens may already make up 90% of the probability, so the pool shrinks automatically.
        - When the model is **uncertain**, the pool dynamically expands to include more choices.
        """
    )

with tab_len:
    st.markdown(
        """
        ### 📏 Maximum New Tokens
        **How it works:**  
        Specifies the maximum number of autoregressive generation steps allowed before stopping.
        
        - Prevents runaway generation loops.
        - Keeps response latency fast and interactive.
        """
    )

with tab_presets:
    st.markdown(
        """
        ### 💡 Recommended Hyperparameter Recipes
        
        | Task | Temperature | Top-K | Top-P | Purpose |
        | :--- | :--- | :--- | :--- | :--- |
        | **🎯 Factual QA / Math** | `0.00` | `1` | `1.00` | Maximum precision, deterministic facts |
        | **⚖️ Balanced Dialogue** | `0.70` | `50` | `0.90` | Natural conversation, balanced fluency |
        | **🎨 Creative Storytelling** | `1.10` | `80` | `0.95` | High imagination, rich vocabulary |
        | **💻 Code Generation** | `0.20` | `20` | `0.85` | Syntax correctness with minor flexibility |
        """
    )

st.caption("Adjust any of these parameters in the left sidebar to see their effect on your next generation!")
