"""
LocalGPT: ChatGPT-Style Conversational Interface
A clean, private, 100% offline conversational AI assistant powered by Qwen/Qwen2.5-1.5B-Instruct.
Includes complete chat history management: New Chat, Open Old Chat, Rename, and Delete.
Step 9: Message Controls — Copy, Regenerate, X-Ray for AI responses; Edit for user messages.
Step 10: Settings Panel — System Instructions, Temperature, Top-K, Top-P, Max Tokens with full validation.
"""

import time
import uuid
from typing import Dict, Any, List, Optional
import streamlit as st

# Local module imports
from model import (
    get_device_info,
    load_model_and_tokenizer,
    stream_chat_response,
    clear_memory_cache,
)
from tokenizer import format_chat_prompt, tokenize_text
from database import (
    save_conversation,
    load_conversation,
    load_all_conversations,
    delete_conversation,
    rename_conversation,
)
from document_loader import (
    load_and_extract_document,
    save_uploaded_file,
    list_saved_documents,
    delete_saved_document,
    get_documents_directory,
)
from rag import LocalRAG, format_answer_with_sources
from xray import (
    extract_embeddings,
    get_transformer_layers_info,
    extract_attentions,
    get_attention_matrix,
    extract_hidden_states,
    get_hidden_state_for_layer,
    extract_next_token_logits,
)
from visualization import (
    plot_embeddings_2d,
    plot_architecture_stack,
    plot_attention_heatmap,
    plot_hidden_states_2d,
    plot_next_token_probabilities,
)

# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="LocalGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished ChatGPT-style aesthetics
st.markdown(
    """
    <style>
    /* Main container styling */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 5rem;
        max-width: 900px;
    }
    /* Welcome hero banner */
    .welcome-container {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
    }
    .welcome-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #00ADB5 0%, #3a86ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .welcome-subtitle {
        color: #888888;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    /* Chat header banner in main area */
    .chat-header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0.8rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .chat-title-text {
        font-size: 1.2rem;
        font-weight: 600;
        color: #f0f2f6;
    }
    .chat-meta-text {
        font-size: 0.8rem;
        color: #718096;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #11141a;
    }
    .sidebar-chat-item {
        margin-bottom: 2px;
    }
    /* Action controls micro-buttons styling */
    .msg-control-bar {
        display: flex;
        gap: 8px;
        align-items: center;
        margin-top: 4px;
        margin-bottom: 4px;
    }
    /* Settings panel styling */
    .settings-header {
        font-size: 0.95rem;
        font-weight: 600;
        color: #00ADB5;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# MODEL & TOKENIZER INITIALIZATION (CACHED)
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Local Qwen-1.5B model into memory...")
def get_model_and_tokenizer():
    return load_model_and_tokenizer()

model, tokenizer = get_model_and_tokenizer()
dev_info = get_device_info()

@st.cache_resource(show_spinner="Initializing Local FAISS RAG Engine...")
def get_rag_engine():
    return LocalRAG(model=model, tokenizer=tokenizer)

rag_engine = get_rag_engine()

# -------------------------------------------------------------
# DEFAULT SETTINGS CONSTANTS
# -------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are LocalGPT, an intelligent, helpful, and concise AI assistant running 100% locally on this device. "
    "Provide clear, accurate, and structured responses."
)
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 512

# -------------------------------------------------------------
# CONVERSATION MANAGEMENT HELPERS
# -------------------------------------------------------------
def generate_unique_chat_id() -> str:
    """Generates a guaranteed unique ID for a conversation session."""
    return f"chat_{int(time.time())}_{uuid.uuid4().hex[:6]}"


def create_new_chat(title: str = "New Chat") -> str:
    """Creates a new conversation session with full metadata and saves to local database."""
    chat_id = generate_unique_chat_id()
    now = time.time()
    chat_data = {
        "id": chat_id,
        "title": title,
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }
    st.session_state.chats[chat_id] = chat_data
    st.session_state.current_chat_id = chat_id
    save_conversation(
        chat_id=chat_id,
        title=title,
        messages=[],
        created_at=now,
        updated_at=now,
    )
    return chat_id


def get_active_chat() -> Dict[str, Any]:
    """Retrieves or creates the currently active chat."""
    if not st.session_state.chats:
        create_new_chat()
    if st.session_state.current_chat_id not in st.session_state.chats:
        sorted_chats = sorted(
            st.session_state.chats.values(),
            key=lambda c: c.get("updated_at", c.get("created_at", 0)),
            reverse=True,
        )
        st.session_state.current_chat_id = sorted_chats[0]["id"]
    return st.session_state.chats[st.session_state.current_chat_id]


def delete_chat(chat_id: str) -> None:
    """Safely deletes a conversation from session state and SQLite database."""
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        delete_conversation(chat_id)
        if not st.session_state.chats:
            create_new_chat()
        elif st.session_state.current_chat_id == chat_id:
            sorted_chats = sorted(
                st.session_state.chats.values(),
                key=lambda c: c.get("updated_at", 0),
                reverse=True,
            )
            st.session_state.current_chat_id = sorted_chats[0]["id"]


def rename_chat(chat_id: str, new_title: str) -> None:
    """Renames a conversation and updates local SQLite database."""
    clean = new_title.strip()
    if chat_id in st.session_state.chats and clean:
        now = time.time()
        st.session_state.chats[chat_id]["title"] = clean
        st.session_state.chats[chat_id]["updated_at"] = now
        rename_conversation(chat_id, clean)


# -------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------
if "chats" not in st.session_state:
    loaded_chats = load_all_conversations()
    if loaded_chats:
        st.session_state.chats = loaded_chats
        sorted_cids = sorted(
            loaded_chats.keys(),
            key=lambda k: loaded_chats[k].get("updated_at", loaded_chats[k].get("created_at", 0)),
            reverse=True,
        )
        st.session_state.current_chat_id = sorted_cids[0]
    else:
        st.session_state.chats = {}
        init_id = create_new_chat("New Chat")
        st.session_state.current_chat_id = init_id

# Step 10: Persist Settings in Session State
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

if "temperature" not in st.session_state:
    st.session_state.temperature = DEFAULT_TEMPERATURE

if "top_k" not in st.session_state:
    st.session_state.top_k = DEFAULT_TOP_K

if "top_p" not in st.session_state:
    st.session_state.top_p = DEFAULT_TOP_P

if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = DEFAULT_MAX_TOKENS

if "rename_dialog_chat_id" not in st.session_state:
    st.session_state.rename_dialog_chat_id = None

if "editing_msg_idx" not in st.session_state:
    st.session_state.editing_msg_idx = None

if "pending_generation" not in st.session_state:
    st.session_state.pending_generation = False

if "app_mode" not in st.session_state:
    st.session_state.app_mode = "💬 Normal Chat"

if "rag_top_k" not in st.session_state:
    st.session_state.rag_top_k = 3

if "active_xray_msg_idx" not in st.session_state:
    st.session_state.active_xray_msg_idx = None

active_chat = get_active_chat()

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.title("🤖 LOCALGPT")
    st.caption("100% Offline & Private LLM")

    # Mode Selector: Normal Chat vs Document RAG
    app_mode = st.radio(
        "Application Mode:",
        ["💬 Normal Chat", "📚 Document QA (RAG)"],
        index=0 if st.session_state.app_mode == "💬 Normal Chat" else 1,
        key="app_mode_radio",
    )
    if app_mode != st.session_state.app_mode:
        st.session_state.app_mode = app_mode
        st.rerun()

    st.markdown("---")

    # 1. ➕ New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        create_new_chat()
        st.session_state.editing_msg_idx = None
        st.session_state.pending_generation = False
        st.session_state.active_xray_msg_idx = None
        st.rerun()

    st.markdown("---")

    # 2. Recent Chats Section
    st.subheader("Recent Chats")
    
    # Sort conversations by updated_at (newest first)
    sorted_chat_list = sorted(
        st.session_state.chats.values(),
        key=lambda c: c.get("updated_at", c.get("created_at", 0)),
        reverse=True,
    )

    for chat_item in sorted_chat_list:
        cid = chat_item["id"]
        title = chat_item.get("title", "New Chat")
        is_active = (cid == st.session_state.current_chat_id)
        msg_count = len(chat_item.get("messages", []))

        # Format label with active indicator and message badge
        if is_active:
            col_btn, col_opt = st.columns([0.78, 0.22])
            with col_btn:
                st.button(
                    f"👉 **{title}**",
                    key=f"active_chat_{cid}",
                    use_container_width=True,
                    help=f"Active Chat ({msg_count} messages)",
                )
            with col_opt:
                # Popover for Rename & Delete actions on active chat
                with st.popover("⚙️", use_container_width=True):
                    st.markdown("**Chat Options**")
                    new_title_input = st.text_input("Rename Title:", value=title, key=f"rename_input_{cid}")
                    if st.button("💾 Save Title", key=f"save_rename_{cid}", use_container_width=True):
                        rename_chat(cid, new_title_input)
                        st.rerun()
                    st.markdown("---")
                    if st.button("🗑️ Delete Chat", key=f"del_chat_{cid}", use_container_width=True, type="secondary"):
                        delete_chat(cid)
                        st.rerun()
        else:
            col_btn, col_del = st.columns([0.82, 0.18])
            with col_btn:
                if st.button(
                    f"💬 {title}",
                    key=f"chat_btn_{cid}",
                    use_container_width=True,
                    help=f"Open chat ({msg_count} messages)",
                ):
                    st.session_state.current_chat_id = cid
                    st.session_state.editing_msg_idx = None
                    st.session_state.pending_generation = False
                    st.session_state.active_xray_msg_idx = None
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"quick_del_{cid}", help="Delete chat", use_container_width=True):
                    delete_chat(cid)
                    st.rerun()

    st.markdown("---")

    # 3. Step 10 Settings Panel
    with st.expander("⚙️ Settings", expanded=False):
        st.markdown("<div class='settings-header'>System Instructions</div>", unsafe_allow_html=True)
        custom_sys = st.text_area(
            "Persona / Instructions:",
            value=st.session_state.system_prompt,
            height=100,
            key="sys_prompt_input",
            help="Defines the AI assistant's persona, formatting rules, and behavior.",
        )
        if custom_sys != st.session_state.system_prompt:
            st.session_state.system_prompt = custom_sys

        st.markdown("<div class='settings-header'>Generation Hyperparameters</div>", unsafe_allow_html=True)
        
        # Temperature Slider (0.0 to 1.5, default 0.7)
        new_temp = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.5,
            value=float(st.session_state.temperature),
            step=0.05,
            key="temp_slider",
            help="Controls randomness. 0.0 = completely deterministic/greedy, 1.0+ = creative/diverse.",
        )
        st.session_state.temperature = new_temp

        # Top-K Slider (1 to 100, default 50)
        new_top_k = st.slider(
            "Top-K",
            min_value=1,
            max_value=100,
            value=int(st.session_state.top_k),
            step=1,
            key="top_k_slider",
            help="Filters logits to keep only top K tokens before sampling.",
        )
        st.session_state.top_k = new_top_k

        # Top-P Slider (0.05 to 1.0, default 0.9)
        new_top_p = st.slider(
            "Top-P (Nucleus)",
            min_value=0.05,
            max_value=1.0,
            value=float(st.session_state.top_p),
            step=0.05,
            key="top_p_slider",
            help="Selects tokens with cumulative probability up to P.",
        )
        st.session_state.top_p = new_top_p

        # Max Tokens Slider (32 to 1024, default 512)
        new_max_tokens = st.slider(
            "Max Tokens",
            min_value=32,
            max_value=1024,
            value=int(st.session_state.max_tokens),
            step=16,
            key="max_tokens_slider",
            help="Maximum number of new tokens to generate per turn (capped for memory safety).",
        )
        st.session_state.max_tokens = new_max_tokens

        # Reset to Defaults button
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
            st.session_state.temperature = DEFAULT_TEMPERATURE
            st.session_state.top_k = DEFAULT_TOP_K
            st.session_state.top_p = DEFAULT_TOP_P
            st.session_state.max_tokens = DEFAULT_MAX_TOKENS
            st.rerun()

    # 4. Step 11: Documents & Uploads Panel
    with st.expander("📄 Documents & Knowledge", expanded=False):
        st.markdown("<div class='settings-header'>Upload Local Documents</div>", unsafe_allow_html=True)
        st.caption("Upload PDF, DOCX, or TXT documents to store in local knowledge (`data/documents`).")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            key="doc_uploader",
            label_visibility="collapsed",
        )
        
        if uploaded_files:
            for up_file in uploaded_files:
                processed_key = f"processed_{up_file.name}_{up_file.size}"
                if processed_key not in st.session_state:
                    with st.spinner(f"Extracting {up_file.name}..."):
                        file_bytes = up_file.read()
                        extracted_doc = load_and_extract_document(
                            file_source=file_bytes,
                            filename=up_file.name,
                            save_to_dir=get_documents_directory(),
                        )
                        st.session_state[processed_key] = True
                        if extracted_doc.is_valid:
                            st.success(
                                f"✅ **{up_file.name}** processed successfully!\n\n"
                                f"*Pages: {extracted_doc.page_count} | Words: {extracted_doc.total_words} | Chars: {extracted_doc.total_chars}*"
                            )
                        else:
                            st.error(f"⚠️ Failed to extract **{up_file.name}**: {extracted_doc.error or 'Invalid file.'}")

        # List saved documents in data/documents
        saved_docs = list_saved_documents()
        st.markdown("---")
        st.markdown(f"<div class='settings-header'>Stored Documents ({len(saved_docs)})</div>", unsafe_allow_html=True)
        
        if not saved_docs:
            st.caption("No documents stored yet. Upload a PDF, DOCX, or TXT above.")
        else:
            for doc_info in saved_docs:
                d_name = doc_info["filename"]
                d_ext = doc_info["extension"].upper()
                d_size = doc_info["size_str"]
                
                icon = "📕" if d_ext == "PDF" else ("📘" if d_ext == "DOCX" else "📝")
                
                col_info, col_del = st.columns([0.82, 0.18])
                with col_info:
                    st.markdown(f"{icon} **{d_name}** (`{d_size}` &bull; `{d_ext}`)")
                with col_del:
                    if st.button("🗑️", key=f"del_doc_{d_name}", help=f"Delete {d_name}", use_container_width=True):
                        delete_saved_document(d_name)
                        st.rerun()
                
                # Document Preview Expander
                with st.expander(f"🔍 Preview {d_name}", expanded=False):
                    doc_extracted = load_and_extract_document(doc_info["file_path"])
                    if doc_extracted.is_valid:
                        st.markdown(f"**Pages:** `{doc_extracted.page_count}` &bull; **Words:** `{doc_extracted.total_words}` &bull; **Chars:** `{doc_extracted.total_chars}`")
                        if doc_extracted.file_type == "pdf" and doc_extracted.pages:
                            for pg in doc_extracted.pages:
                                st.markdown(f"**Page {pg['page_number']}** ({pg['word_count']} words):")
                                st.text(pg["text"][:300] + ("..." if len(pg["text"]) > 300 else ""))
                        else:
                            st.text(doc_extracted.full_text[:400] + ("..." if len(doc_extracted.full_text) > 400 else ""))
                    else:
                        st.warning(f"Preview unavailable: {doc_extracted.error}")

        # FAISS Index Status & Trigger
        st.markdown("---")
        rag_stats = rag_engine.vector_store.get_stats()
        st.markdown("<div class='settings-header'>FAISS Vector Index</div>", unsafe_allow_html=True)
        st.caption(f"⚡ **Indexed:** `{rag_stats['total_chunks']}` chunks &bull; `{rag_stats['total_documents']}` files &bull; Dim: `{rag_stats['embedding_dim']}`")
        if st.button("⚡ Index Documents into FAISS", use_container_width=True, help="Compute embeddings and build FAISS vector index"):
            with st.spinner("Embedding documents into FAISS..."):
                count = rag_engine.index_directory()
            st.success(f"Indexed {count} chunks into FAISS!")
            st.rerun()

    # 5. Hardware status badge
    st.markdown("---")
    if dev_info["is_gpu"]:
        st.success(f"⚡ **GPU:** {dev_info['name']}\n\n*{dev_info['memory_details']}*")
    else:
        st.info(f"💻 **CPU Mode:** {dev_info['name']}\n\n*Running locally on system RAM*")

# -------------------------------------------------------------
# MAIN AREA
# -------------------------------------------------------------
active_chat = get_active_chat()
chat_messages = active_chat.get("messages", [])

# Chat Header Bar with title and quick actions
col_title, col_actions = st.columns([0.72, 0.28])
with col_title:
    st.markdown(f"### 💬 {active_chat.get('title', 'New Chat')}")
    created_time_str = time.strftime("%b %d, %H:%M", time.localtime(active_chat.get("created_at", time.time())))
    updated_time_str = time.strftime("%b %d, %H:%M", time.localtime(active_chat.get("updated_at", time.time())))
    st.caption(f"Created: {created_time_str} &bull; Updated: {updated_time_str} &bull; Messages: {len(chat_messages)}")

with col_actions:
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        with st.popover("✏️ Rename", use_container_width=True):
            st.markdown("**Rename Conversation**")
            header_new_title = st.text_input("Title:", value=active_chat.get("title", ""), key="header_rename_val")
            if st.button("Save", key="header_save_title_btn", use_container_width=True):
                rename_chat(active_chat["id"], header_new_title)
                st.rerun()
    with col_a2:
        if st.button("🗑️ Delete", key="header_del_chat_btn", use_container_width=True, help="Delete this conversation"):
            delete_chat(active_chat["id"])
            st.rerun()

st.markdown("---")

# Welcome hero when conversation is empty
if len(chat_messages) == 0 and not st.session_state.pending_generation:
    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-title">How can I help you today?</div>
            <div class="welcome-subtitle">Powered locally by Qwen/Qwen2.5-1.5B-Instruct &bull; Private &bull; No API Keys</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick starter prompt suggestions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💡 Explain quantum computing simply", use_container_width=True):
            st.session_state.starter_prompt = "Explain quantum computing simply in 2-3 short paragraphs."
        if st.button("🐍 Write a Python script to sort a list", use_container_width=True):
            st.session_state.starter_prompt = "Write a clean Python function to sort a list of dictionaries by a key."
    with col2:
        if st.button("🧠 What is artificial intelligence?", use_container_width=True):
            st.session_state.starter_prompt = "What is artificial intelligence?"
        if st.button("📝 Summarize the benefits of local LLMs", use_container_width=True):
            st.session_state.starter_prompt = "Summarize the key benefits of running local LLMs on device."

# -------------------------------------------------------------
# DISPLAY CONVERSATION MESSAGE HISTORY WITH STEP 9 CONTROLS
# -------------------------------------------------------------
for idx, msg in enumerate(chat_messages):
    role = msg.get("role")
    content = msg.get("content", "")

    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
            
            # Step 9 User Message Control: Edit
            is_editing = (st.session_state.editing_msg_idx == idx)
            
            if is_editing:
                st.markdown("---")
                edited_input = st.text_area(
                    "✏️ Edit your message:",
                    value=content,
                    key=f"edit_text_area_{idx}",
                    height=100,
                )
                col_save, col_cancel, _ = st.columns([0.28, 0.22, 0.5])
                with col_save:
                    if st.button("💾 Resubmit", key=f"save_edit_btn_{idx}", type="primary", use_container_width=True):
                        clean_text = edited_input.strip()
                        if clean_text:
                            # 1. Update user message content at idx
                            active_chat["messages"][idx]["content"] = clean_text
                            # 2. Truncate conversation history after this user message
                            active_chat["messages"] = active_chat["messages"][:idx + 1]
                            active_chat["updated_at"] = time.time()
                            # 3. If first message, update chat title
                            if idx == 0:
                                first_line = clean_text.replace("\n", " ").strip()
                                active_chat["title"] = (first_line[:28] + "...") if len(first_line) > 28 else first_line
                            # Save state before regenerating response
                            save_conversation(
                                chat_id=active_chat["id"],
                                title=active_chat.get("title", "New Chat"),
                                messages=active_chat["messages"],
                                created_at=active_chat.get("created_at"),
                                updated_at=active_chat["updated_at"],
                            )
                            # 4. Trigger streaming generation for the updated context
                            st.session_state.editing_msg_idx = None
                            st.session_state.pending_generation = True
                            st.rerun()
                with col_cancel:
                    if st.button("❌ Cancel", key=f"cancel_edit_btn_{idx}", use_container_width=True):
                        st.session_state.editing_msg_idx = None
                        st.rerun()
            else:
                col_edit_btn, _ = st.columns([0.15, 0.85])
                with col_edit_btn:
                    if st.button("✏️ Edit", key=f"edit_trigger_btn_{idx}", help="Edit this message and resubmit", use_container_width=True):
                        st.session_state.editing_msg_idx = idx
                        st.rerun()

    elif role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(content)
            
            # Display RAG sources if present
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"📚 Retrieved Context Sources ({len(sources)} Chunks)", expanded=False):
                    for s in sources:
                        pg_label = f"Page {s.get('page_number', 1)}" if s.get("page_number") else "Page 1"
                        st.markdown(f"**[{s.get('rank', 1)}] {s.get('source')}** &bull; `{pg_label}` &bull; *Relevance:* `{s.get('score_pct', '')}`")
                        st.caption(s.get("text", ""))

            # AI Message Controls: Copy, Regenerate, and Advanced Neural X-Ray
            ctrl_col1, ctrl_col2, ctrl_col3, _ = st.columns([0.16, 0.22, 0.22, 0.40])
            
            # 1. 📋 COPY CONTROL
            with ctrl_col1:
                with st.popover("📋 Copy", use_container_width=True, help="Copy complete AI response"):
                    st.markdown("**📋 Complete Response Text**")
                    st.text_area(
                        "Raw Text (Click to select & copy):",
                        value=content,
                        height=140,
                        key=f"copy_raw_area_{idx}",
                        label_visibility="collapsed",
                    )
                    st.caption("✅ The full response text is ready for copying.")
            
            # 2. 🔄 REGENERATE CONTROL
            with ctrl_col2:
                if st.button("🔄 Regenerate", key=f"regen_btn_{idx}", help="Regenerate this response using same context", use_container_width=True):
                    active_chat["messages"] = active_chat["messages"][:idx]
                    active_chat["updated_at"] = time.time()
                    save_conversation(
                        chat_id=active_chat["id"],
                        title=active_chat.get("title", "New Chat"),
                        messages=active_chat["messages"],
                        created_at=active_chat.get("created_at"),
                        updated_at=active_chat["updated_at"],
                    )
                    st.session_state.editing_msg_idx = None
                    st.session_state.active_xray_msg_idx = None
                    st.session_state.pending_generation = True
                    st.rerun()

            # 3. 🔬 X-RAY ADVANCED INSPECTION CONTROL (STEP 15-18)
            is_xray_open = (st.session_state.get("active_xray_msg_idx") == idx)
            with ctrl_col3:
                xray_btn_text = "🔬 Hide X-Ray" if is_xray_open else "🔬 X-Ray"
                if st.button(
                    xray_btn_text,
                    key=f"xray_toggle_btn_{idx}",
                    use_container_width=True,
                    help="Toggle advanced inspection for tokens, embeddings, layers, attention, hidden states, logits, and probabilities",
                    type="primary" if is_xray_open else "secondary",
                ):
                    st.session_state.active_xray_msg_idx = None if is_xray_open else idx
                    st.rerun()

            # Step 15: Interactive Neural X-Ray Inspection Suite
            if is_xray_open:
                st.markdown("---")
                col_xhdr, col_xclose = st.columns([0.82, 0.18])
                with col_xhdr:
                    st.markdown(f"#### 🔬 Neural X-Ray Inspection &bull; Turn #{idx // 2 + 1}")
                    st.caption("Inspecting actual tensor values and representations from `Qwen/Qwen2.5-1.5B-Instruct`.")
                with col_xclose:
                    if st.button("❌ Close", key=f"close_xray_btn_{idx}", use_container_width=True, help="Close X-Ray inspection"):
                        st.session_state.active_xray_msg_idx = None
                        st.rerun()

                # User prompt for this turn
                user_prompt_text = ""
                if idx > 0 and active_chat["messages"][idx - 1].get("role") == "user":
                    user_prompt_text = active_chat["messages"][idx - 1].get("content", "")

                try:
                    tok_data = tokenize_text(content, tokenizer=tokenizer, max_length=256)
                    prompt_tok_data = tokenize_text(user_prompt_text, tokenizer=tokenizer, max_length=128) if user_prompt_text else {"tokens": [], "token_ids": [], "total_tokens": 0, "breakdown": []}
                    
                    tab_tok, tab_emb, tab_layers, tab_attn, tab_hidden, tab_logits = st.tabs([
                        "🔤 Tokens",
                        "🌌 Embeddings",
                        "🏛️ Layers",
                        "🎯 Attention",
                        "🧠 Hidden States",
                        "📈 Logits",
                    ])

                    # TAB 1: TOKENS (Tokenized Input & Response)
                    with tab_tok:
                        st.markdown("##### 🔤 Tokenized Input & Output Sequence")
                        st.caption("Subword decomposition and vocabulary indices produced by the Qwen BPE Tokenizer.")
                        
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        with col_m1:
                            st.metric("Prompt Tokens", prompt_tok_data["total_tokens"])
                        with col_m2:
                            st.metric("Response Tokens", tok_data["total_tokens"])
                        with col_m3:
                            st.metric("Total Tokens", prompt_tok_data["total_tokens"] + tok_data["total_tokens"])
                        with col_m4:
                            vocab_sz = getattr(tokenizer, "vocab_size", 151936)
                            st.metric("Vocab Size", f"{vocab_sz:,}")

                        if user_prompt_text and prompt_tok_data["breakdown"]:
                            with st.expander(f"📥 Tokenized Input Prompt ({prompt_tok_data['total_tokens']} tokens)", expanded=True):
                                prompt_df = [
                                    {
                                        "Index": item["index"] + 1,
                                        "Token": item["token"],
                                        "Display": item["token_display"],
                                        "Token ID": item["token_id"],
                                        "Chars": len(item["token"]),
                                    }
                                    for item in prompt_tok_data["breakdown"]
                                ]
                                st.dataframe(prompt_df, use_container_width=True, height=180)

                        if tok_data["breakdown"]:
                            with st.expander(f"📤 Tokenized AI Response ({tok_data['total_tokens']} tokens)", expanded=True):
                                tok_df = [
                                    {
                                        "Index": item["index"] + 1,
                                        "Token": item["token"],
                                        "Display": item["token_display"],
                                        "Token ID": item["token_id"],
                                        "Chars": len(item["token"]),
                                    }
                                    for item in tok_data["breakdown"][:64]
                                ]
                                st.dataframe(tok_df, use_container_width=True, height=200)
                                if len(tok_data["breakdown"]) > 64:
                                    st.caption(f"Displaying first 64 of {tok_data['total_tokens']} response tokens.")

                    # TAB 2: EMBEDDINGS (Token ID, Dimension, Statistics, PCA)
                    with tab_emb:
                        st.markdown("##### 🌌 Input Embedding Layer (1536-Dimensional Space)")
                        st.caption("Real 1536-dimensional dense vectors extracted directly from Qwen's input embedding matrix.")
                        
                        if tok_data["token_ids"]:
                            emb_tok_ids = tok_data["token_ids"][:36]
                            emb_toks = tok_data["tokens"][:36]
                            emb_res = extract_embeddings(emb_tok_ids, model=model)
                            
                            if emb_res.get("error"):
                                st.error(emb_res["error"])
                            else:
                                stats = emb_res.get("global_stats", {})
                                col_e1, col_e2, col_e3, col_e4 = st.columns(4)
                                with col_e1:
                                    st.metric("Embedding Dim", stats.get("embedding_dim", 1536))
                                with col_e2:
                                    st.metric("Mean Vector Norm", f"{stats.get('mean_norm', 0.0):.4f}")
                                with col_e3:
                                    st.metric("Global Mean", f"{stats.get('global_mean', 0.0):.4f}")
                                with col_e4:
                                    st.metric("Global Std", f"{stats.get('global_std', 0.0):.4f}")

                                fig_pca = plot_embeddings_2d(
                                    emb_toks,
                                    emb_tok_ids,
                                    emb_res["embeddings_matrix"],
                                )
                                st.plotly_chart(fig_pca, use_container_width=True)

                                with st.expander("🔍 Token Embedding Vector Table (Token ID, Norm, & 8-D Head)", expanded=False):
                                    emb_table = [
                                        {
                                            "Index": item["index"] + 1,
                                            "Token": emb_toks[item["index"]] if item["index"] < len(emb_toks) else "",
                                            "Token ID": item["token_id"],
                                            "L2 Norm": f"{item['norm']:.4f}",
                                            "Vector Head (First 8-D)": str([round(x, 4) for x in item["vector_preview"]]),
                                        }
                                        for item in emb_res.get("token_embeddings", [])[:25]
                                    ]
                                    st.dataframe(emb_table, use_container_width=True, height=200)

                    # TAB 3: LAYERS (Architecture Stack & Details)
                    with tab_layers:
                        st.markdown("##### 🏛️ 28-Layer Transformer Architecture")
                        st.caption("Structural hierarchy, parameters, and attention mechanisms of Qwen/Qwen2.5-1.5B-Instruct.")
                        
                        layers_info = get_transformer_layers_info(model=model)
                        num_layers = layers_info.get("num_layers", 28)
                        
                        col_l1, col_l2, col_l3, col_l4 = st.columns(4)
                        with col_l1:
                            st.metric("Total Layers", num_layers)
                        with col_l2:
                            st.metric("Hidden Size", layers_info.get("hidden_size", 1536))
                        with col_l3:
                            st.metric("Attention Heads", f"{layers_info.get('num_attention_heads', 12)} Q / {layers_info.get('num_key_value_heads', 2)} KV")
                        with col_l4:
                            total_p = layers_info.get("total_parameters", 1540000000)
                            st.metric("Parameters", f"{total_p / 1e9:.2f}B" if total_p else "1.54B")

                        selected_layer = st.slider(
                            "Select Layer to Inspect:",
                            min_value=1,
                            max_value=num_layers,
                            value=1,
                            key=f"layer_slider_{idx}",
                        )
                        
                        fig_arch = plot_architecture_stack(num_layers, selected_layer)
                        st.plotly_chart(fig_arch, use_container_width=True)

                    # TAB 4: ATTENTION (Layer & Head Selection + Heatmap)
                    with tab_attn:
                        st.markdown("##### 🎯 Multi-Head Self-Attention Weights")
                        st.caption("Real attention weight matrices computed by Qwen's attention heads during forward execution.")
                        
                        if tok_data["token_ids"]:
                            attn_tok_ids = tok_data["token_ids"][:20]
                            attn_toks = tok_data["tokens"][:20]
                            
                            with st.spinner("Computing real attention weights..."):
                                attn_data = extract_attentions(attn_tok_ids, model=model)
                            
                            if attn_data.get("error"):
                                st.error(attn_data["error"])
                            else:
                                col_c1, col_c2 = st.columns(2)
                                with col_c1:
                                    sel_attn_layer = st.slider(
                                        "Select Layer:",
                                        min_value=1,
                                        max_value=attn_data.get("num_layers", 28),
                                        value=1,
                                        key=f"attn_layer_slider_{idx}",
                                    )
                                with col_c2:
                                    head_opts = ["Average Across All Heads"] + [f"Head {h}" for h in range(1, attn_data.get("num_heads", 12) + 1)]
                                    sel_head_str = st.selectbox(
                                        "Select Head:",
                                        options=head_opts,
                                        index=0,
                                        key=f"attn_head_select_{idx}",
                                    )
                                
                                is_avg = (sel_head_str == "Average Across All Heads")
                                sel_head_idx = None if is_avg else int(sel_head_str.split(" ")[1]) - 1
                                
                                attn_matrix = get_attention_matrix(attn_data, sel_attn_layer - 1, sel_head_idx)
                                fig_attn = plot_attention_heatmap(
                                    tokens=attn_toks,
                                    attention_matrix=attn_matrix,
                                    layer_num=sel_attn_layer,
                                    head_num=(sel_head_idx + 1) if sel_head_idx is not None else None,
                                    is_average=is_avg,
                                )
                                st.plotly_chart(fig_attn, use_container_width=True)

                    # TAB 5: HIDDEN STATES (Layer Selection, Numerical Info, PCA)
                    with tab_hidden:
                        st.markdown("##### 🧠 Intermediate Hidden States (Layers 0..28)")
                        st.caption("Inspect real representation vectors across all 28 Transformer layers and pre-transformer embeddings.")
                        
                        if tok_data["token_ids"]:
                            hidden_tok_ids = tok_data["token_ids"][:20]
                            hidden_toks = tok_data["tokens"][:20]
                            
                            with st.spinner("Extracting intermediate hidden states..."):
                                hidden_data = extract_hidden_states(hidden_tok_ids, model=model)
                            
                            if hidden_data.get("error"):
                                st.error(hidden_data["error"])
                            else:
                                sel_hidden_layer = st.slider(
                                    "Select Layer (0 = Input Embeddings, 1..28 = Transformer Layers):",
                                    min_value=0,
                                    max_value=hidden_data.get("num_layers", 28),
                                    value=1,
                                    key=f"hidden_layer_slider_{idx}",
                                )
                                
                                layer_state = get_hidden_state_for_layer(hidden_data, sel_hidden_layer)
                                if layer_state:
                                    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
                                    with col_h1:
                                        st.metric("Hidden Dim", layer_state.get("hidden_dim", 1536))
                                    with col_h2:
                                        st.metric("Mean Activation", f"{layer_state.get('mean', 0.0):.4f}")
                                    with col_h3:
                                        st.metric("Std Dev", f"{layer_state.get('std', 0.0):.4f}")
                                    with col_h4:
                                        st.metric("Mean L2 Norm", f"{layer_state.get('mean_l2_norm', 0.0):.4f}")

                                    fig_hidden = plot_hidden_states_2d(
                                        tokens=hidden_toks,
                                        token_ids=hidden_tok_ids,
                                        hidden_matrix=layer_state["matrix"],
                                        layer_label=layer_state["name"],
                                        layer_num=sel_hidden_layer,
                                    )
                                    st.plotly_chart(fig_hidden, use_container_width=True)

                                    with st.expander(f"📊 {layer_state['name']} Token Vector Norms", expanded=False):
                                        norms_df = [
                                            {
                                                "Position": p_idx + 1,
                                                "Token": hidden_toks[p_idx] if p_idx < len(hidden_toks) else "",
                                                "Token ID": hidden_tok_ids[p_idx] if p_idx < len(hidden_tok_ids) else "",
                                                "L2 Norm": f"{layer_state['token_norms'][p_idx]:.4f}" if p_idx < len(layer_state.get("token_norms", [])) else "",
                                            }
                                            for p_idx in range(len(hidden_toks))
                                        ]
                                        st.dataframe(norms_df, use_container_width=True, height=180)

                    # TAB 6: NEXT TOKEN PREDICTION & LOGITS (STEP 17)
                    with tab_logits:
                        st.markdown("##### 🎯 Next Token Prediction")
                        st.caption("Inspect the highest-probability candidate tokens and softmax probabilities calculated from actual model logits at any sequence position.")
                        
                        if tok_data["token_ids"]:
                            seq_len_avail = min(len(tok_data["token_ids"]), 48)
                            col_ctrl_pos, col_ctrl_k = st.columns([0.6, 0.4])
                            with col_ctrl_pos:
                                selected_pos = st.slider(
                                    "Inspect Position in Sequence:",
                                    min_value=1,
                                    max_value=seq_len_avail,
                                    value=seq_len_avail,
                                    key=f"logit_pos_slider_{idx}",
                                    help="Select which token position to compute next-token candidate predictions for.",
                                )
                            with col_ctrl_k:
                                num_candidates = st.slider(
                                    "Candidates to Display:",
                                    min_value=3,
                                    max_value=20,
                                    value=10,
                                    key=f"num_cand_slider_{idx}",
                                    help="Configure how many top-ranking candidate tokens are shown.",
                                )
                            
                            sub_token_ids = tok_data["token_ids"][:selected_pos]
                            current_tok_str = tok_data["tokens"][selected_pos - 1] if selected_pos <= len(tok_data["tokens"]) else ""
                            
                            with st.spinner("Calculating next-token candidate probabilities..."):
                                logits_res = extract_next_token_logits(sub_token_ids, model=model, tokenizer=tokenizer, top_k=num_candidates)
                            
                            if logits_res.get("error"):
                                st.error(logits_res["error"])
                            else:
                                top_preds = logits_res.get("top_predictions", [])
                                if top_preds:
                                    col_lg1, col_lg2, col_lg3, col_lg4 = st.columns(4)
                                    with col_lg1:
                                        st.metric("Position Token", f"'{current_tok_str}'", f"Pos {selected_pos}/{seq_len_avail}")
                                    with col_lg2:
                                        st.metric("Top-1 Prediction", f"'{top_preds[0]['token']}'", top_preds[0]["probability_pct_str"])
                                    with col_lg3:
                                        st.metric("Predictive Entropy", f"{logits_res.get('entropy', 0.0):.4f}")
                                    with col_lg4:
                                        st.metric("Logits Range", f"[{logits_res.get('logits_min', 0.0):.1f}, {logits_res.get('logits_max', 0.0):.1f}]")

                                    st.markdown("###### 🏆 Top Predicted Candidate Tokens:")
                                    # Formatted Next Token Prediction cards / list (Example format: "is" — 32%)
                                    cand_cols = st.columns(min(len(top_preds), 4))
                                    for c_i, cand in enumerate(top_preds[:8]):
                                        with cand_cols[c_i % min(len(top_preds), 4)]:
                                            st.info(f'**`"{cand["token"]}"`** &mdash; **{cand["probability_pct"]:.1f}%**')

                                    fig_prob = plot_next_token_probabilities(
                                        top_preds,
                                        title_suffix=f"Pos {selected_pos} ('{current_tok_str}')",
                                    )
                                    st.plotly_chart(fig_prob, use_container_width=True)

                                    with st.expander(f"📊 Next Token Candidate Rankings Table (Top {len(top_preds)})", expanded=False):
                                        pred_table = [
                                            {
                                                "Rank": p["rank"],
                                                "Candidate Token": p["token_display"],
                                                "Raw Repr": p["token_repr"],
                                                "Token ID": p["token_id"],
                                                "Probability (%)": p["probability_pct_str"],
                                                "Raw Logit": f"{p['logit']:.4f}",
                                            }
                                            for p in top_preds
                                        ]
                                        st.dataframe(pred_table, use_container_width=True, height=200)

                    clear_memory_cache()

                except Exception as xray_err:
                    st.error(f"⚠️ X-Ray inspection error: {str(xray_err)}")

# -------------------------------------------------------------
# PENDING GENERATION HANDLER (FOR REGENERATE & EDIT RESUBMIT)
# -------------------------------------------------------------
if st.session_state.pending_generation:
    st.session_state.pending_generation = False
    
    if len(active_chat["messages"]) > 0 and active_chat["messages"][-1]["role"] == "user":
        latest_user_text = active_chat["messages"][-1]["content"]
        is_rag_active = (st.session_state.app_mode == "📚 Document QA (RAG)")
        
        with st.chat_message("assistant"):
            if is_rag_active:
                if rag_engine.vector_store.index.ntotal == 0:
                    with st.spinner("Indexing local documents into FAISS..."):
                        rag_engine.index_directory()

                try:
                    stream, sources = rag_engine.stream_answer_query(
                        query=latest_user_text,
                        top_k=int(st.session_state.rag_top_k),
                        temperature=float(st.session_state.temperature),
                        max_new_tokens=int(st.session_state.max_tokens),
                    )
                    if sources:
                        with st.expander(f"📚 Retrieved Context Sources ({len(sources)} Chunks)", expanded=True):
                            for s in sources:
                                pg_label = f"Page {s.get('page_number', 1)}" if s.get("page_number") else "Page 1"
                                st.markdown(f"**[{s.get('rank', 1)}] {s.get('source')}** &bull; `{pg_label}` &bull; *Relevance:* `{s.get('score_pct', '')}`")
                                st.caption(s.get("text", "")[:300] + ("..." if len(s.get("text", "")) > 300 else ""))
                    full_response = st.write_stream(stream)
                except Exception as e:
                    full_response = f"⚠️ RAG Generation error: {str(e)}"
                    sources = []
                    st.error(full_response)
            else:
                sources = []
                messages_payload = [{"role": "system", "content": st.session_state.system_prompt}]
                history_to_include = active_chat["messages"][-20:] if len(active_chat["messages"]) > 20 else active_chat["messages"]
                messages_payload.extend(history_to_include)

                formatted_prompt = format_chat_prompt(messages_payload, tokenizer=tokenizer)

                try:
                    stream = stream_chat_response(
                        formatted_prompt=formatted_prompt,
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=int(st.session_state.max_tokens),
                        temperature=float(st.session_state.temperature),
                        top_p=float(st.session_state.top_p),
                        top_k=int(st.session_state.top_k),
                        do_sample=(float(st.session_state.temperature) > 0.0),
                    )
                    full_response = st.write_stream(stream)
                    if not full_response:
                        full_response = ""
                except Exception as e:
                    full_response = f"⚠️ Generation error: {str(e)}"
                    st.error(full_response)

            if full_response and full_response.strip():
                now_ts = time.time()
                final_answer_text = format_answer_with_sources(full_response.strip(), sources) if is_rag_active else full_response.strip()
                active_chat["messages"].append({
                    "role": "assistant",
                    "content": final_answer_text,
                    "sources": sources if is_rag_active else [],
                })
                active_chat["updated_at"] = now_ts
                save_conversation(
                    chat_id=active_chat["id"],
                    title=active_chat.get("title", "New Chat"),
                    messages=active_chat["messages"],
                    created_at=active_chat.get("created_at"),
                    updated_at=now_ts,
                )

        clear_memory_cache()
        st.rerun()

# -------------------------------------------------------------
# BOTTOM BAR: QUICK UPLOAD & CHAT INPUT (STEP 18)
# -------------------------------------------------------------
col_b_mode, col_b_upload = st.columns([0.80, 0.20])
with col_b_mode:
    if st.session_state.app_mode == "📚 Document QA (RAG)":
        st.caption("📚 **Document QA Mode Active** &bull; Answers grounded in local knowledge base (`data/documents`).")
    else:
        st.caption("💬 **Chat Mode Active** &bull; Direct conversation with `Qwen/Qwen2.5-1.5B-Instruct`.")

with col_b_upload:
    with st.popover("📎 Upload", use_container_width=True, help="Quickly upload documents for RAG QA"):
        st.markdown("**📄 Upload Document**")
        st.caption("Upload PDF, DOCX, TXT, or MD files.")
        bottom_up_files = st.file_uploader(
            "Upload files",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            key="bottom_file_uploader",
            label_visibility="collapsed",
        )
        if bottom_up_files:
            for b_file in bottom_up_files:
                b_proc_key = f"bottom_proc_{b_file.name}_{b_file.size}"
                if b_proc_key not in st.session_state:
                    with st.spinner(f"Extracting {b_file.name}..."):
                        b_doc = load_and_extract_document(
                            file_source=b_file.read(),
                            filename=b_file.name,
                            save_to_dir=get_documents_directory(),
                        )
                        st.session_state[b_proc_key] = True
                        if b_doc.is_valid:
                            st.success(f"✅ {b_file.name} saved! Re-indexing FAISS...")
                            rag_engine.index_directory()
                            st.rerun()
                        else:
                            st.error(f"Failed to process {b_file.name}")

initial_input = st.session_state.pop("starter_prompt", None)
placeholder_txt = "Ask anything about your documents..." if st.session_state.app_mode == "📚 Document QA (RAG)" else "Ask anything..."
user_query = st.chat_input(placeholder_txt) or initial_input

if user_query and user_query.strip() and not st.session_state.pending_generation:
    query_text = user_query.strip()
    now_time = time.time()
    is_rag_active = (st.session_state.app_mode == "📚 Document QA (RAG)")

    # 1. Update Chat Title on first message
    if active_chat.get("title") == "New Chat" and len(active_chat["messages"]) == 0:
        first_line = query_text.replace("\n", " ").strip()
        prefix = "📚 " if is_rag_active else ""
        active_chat["title"] = prefix + ((first_line[:24] + "...") if len(first_line) > 24 else first_line)

    # 2. Append & Display User Message
    active_chat["messages"].append({"role": "user", "content": query_text})
    active_chat["updated_at"] = now_time
    save_conversation(
        chat_id=active_chat["id"],
        title=active_chat.get("title", "New Chat"),
        messages=active_chat["messages"],
        created_at=active_chat.get("created_at"),
        updated_at=now_time,
    )

    with st.chat_message("user"):
        st.markdown(query_text)

    # 3. Generate & Stream AI Response
    with st.chat_message("assistant"):
        if is_rag_active:
            if rag_engine.vector_store.index.ntotal == 0:
                with st.spinner("Indexing local documents into FAISS..."):
                    rag_engine.index_directory()

            try:
                stream, sources = rag_engine.stream_answer_query(
                    query=query_text,
                    top_k=int(st.session_state.rag_top_k),
                    temperature=float(st.session_state.temperature),
                    max_new_tokens=int(st.session_state.max_tokens),
                )
                if sources:
                    with st.expander(f"📚 Retrieved Context Sources ({len(sources)} Chunks)", expanded=True):
                        for s in sources:
                            pg_label = f"Page {s.get('page_number', 1)}" if s.get("page_number") else "Page 1"
                            st.markdown(f"**[{s.get('rank', 1)}] {s.get('source')}** &bull; `{pg_label}` &bull; *Relevance:* `{s.get('score_pct', '')}`")
                            st.caption(s.get("text", "")[:300] + ("..." if len(s.get("text", "")) > 300 else ""))
                full_response = st.write_stream(stream)
            except Exception as e:
                full_response = f"⚠️ RAG Generation error: {str(e)}"
                sources = []
                st.error(full_response)
        else:
            sources = []
            messages_payload = [{"role": "system", "content": st.session_state.system_prompt}]
            history_to_include = active_chat["messages"][-20:] if len(active_chat["messages"]) > 20 else active_chat["messages"]
            messages_payload.extend(history_to_include)

            formatted_prompt = format_chat_prompt(messages_payload, tokenizer=tokenizer)

            try:
                stream = stream_chat_response(
                    formatted_prompt=formatted_prompt,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=int(st.session_state.max_tokens),
                    temperature=float(st.session_state.temperature),
                    top_p=float(st.session_state.top_p),
                    top_k=int(st.session_state.top_k),
                    do_sample=(float(st.session_state.temperature) > 0.0),
                )
                full_response = st.write_stream(stream)
                if not full_response:
                    full_response = ""
            except Exception as e:
                full_response = f"⚠️ Generation error: {str(e)}"
                st.error(full_response)

        # 4. Append Assistant Response to Chat History & update timestamp
        if full_response and full_response.strip():
            final_answer_text = format_answer_with_sources(full_response.strip(), sources) if is_rag_active else full_response.strip()
            now_ts = time.time()
            active_chat["messages"].append({
                "role": "assistant",
                "content": final_answer_text,
                "sources": sources if is_rag_active else [],
            })
            active_chat["updated_at"] = now_ts
            save_conversation(
                chat_id=active_chat["id"],
                title=active_chat.get("title", "New Chat"),
                messages=active_chat["messages"],
                created_at=active_chat.get("created_at"),
                updated_at=now_ts,
            )

    clear_memory_cache()
    st.rerun()
