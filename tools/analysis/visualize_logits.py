#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Visualize the logit distributions of the target and draft models."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
from transformers import AutoTokenizer
import re
import matplotlib

CJK_FONT_CANDIDATES = [
    'Noto Sans CJK SC',
    'WenQuanYi Zen Hei',
    'WenQuanYi Micro Hei',
    'SimHei',

]

try:
    available_fonts = {f.name for f in matplotlib.font_manager.fontManager.ttflist}

except Exception as e:
    print(f"Warning: Could not list Matplotlib fonts: {e}. will try the defaults.")
    available_fonts = set()

use_chinese = False
selected_font = None
for font_name in CJK_FONT_CANDIDATES:
    if font_name in available_fonts:
        selected_font = font_name
        break

if selected_font:
    try:
        plt.rcParams['font.family'] = 'sans-serif' # The font family must be set first
        plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        use_chinese = True
        print(f"--- Chinese font configured successfully: Matplotlib will prefer '{selected_font}' ---")

        # fig_test, ax_test = plt.subplots(figsize=(1,0.5))

        # plt.close(fig_test)

    except Exception as e:
        print(f"Trying to configure a Chinese font '{selected_font}' failed: {e}")
        use_chinese = False

        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

if not use_chinese:
    print("--- Warning: Could not configure a Chinese font. Chinese text in plots may not display correctly, will try English labels.---")

    if 'DejaVu Sans' not in plt.rcParams['font.sans-serif']:
         plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

def clean_token_text(token_text, token_id=None):
    """Replace non-ASCII text with readable ASCII notation and include the token ID."""
    token_text = token_text.strip()

    if not token_text or token_text.isspace():
        return f"ID:{token_id}" if token_id is not None else "EmptyToken"

    try:
        if isinstance(token_text, bytes):
            token_text = token_text.decode('utf-8', 'replace')
    except UnicodeDecodeError:
        return f"DecodeErr[{token_id}]" if token_id is not None else "DecodeError"

    is_mostly_ascii_or_printable = all(32 <= ord(char) < 127 or char.isspace() for char in token_text)

    # has_complex_chars = bool(re.search(r'[^\w\s\d.,!?"\'()-]', token_text))

    is_special_marker = bool(re.match(r"<0x[0-9A-Fa-f]+>", token_text))

    display_text = token_text

    if use_chinese:

        if len(token_text) > 6:

            if any('\u4e00' <= char <= '\u9fff' for char in token_text):
                 if len(token_text) > 4:
                    display_text = token_text[:3] + "…"
            else:
                display_text = token_text[:5] + "…"

    else:
        if is_special_marker or not is_mostly_ascii_or_printable:

            return f"ID:{token_id}" if token_id is not None else f"Token-{hash(token_text) % 1000:03d}"
        else:
            if len(token_text) > 6:
                display_text = token_text[:5] + "…"

            display_text = f"'{display_text}'"

    return f"{display_text}[{token_id}]" if token_id is not None else display_text

def read_tensor_from_file(filename):
    """Read saved tensor data from a file."""
    if not os.path.exists(filename):
        print(f"File does not exist: {filename}")
        return None

    with open(filename, 'rb') as f:
        ndim = np.fromfile(f, dtype=np.uint64, count=1)
        if not ndim.size: return None
        ndim = ndim[0]

        shape = np.fromfile(f, dtype=np.uint64, count=ndim)
        if shape.size != ndim: return None

        data = np.fromfile(f, dtype=np.float32)

        expected_size = np.prod(shape)
        if data.size == expected_size and data.size > 0 :
            data = data.reshape(shape)
            return data
        else:
            print(f"File data size and shape mismatched or empty: {filename}, data size: {data.size}, expected: {expected_size}")
            return None

def apply_softmax(logits):
    """Convert logits to a probability distribution with softmax."""
    if logits is None or logits.size == 0:
        return np.array([])

    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    sum_exp_logits = np.sum(exp_logits, axis=-1, keepdims=True)

    return exp_logits / np.where(sum_exp_logits == 0, 1, sum_exp_logits)

def visualize_top_tokens(target_probs, draft_probs, tokenizer, top_k=10, token_pos=0):
    """Compare the target and draft models using their top-k token probabilities."""
    if target_probs.size == 0 or draft_probs.size == 0:
        print(f"Token Position {token_pos} probability data is empty, Skipping visualization.")
        return

    target_flat = target_probs.flatten()
    draft_flat = draft_probs.flatten()

    effective_top_k_target = min(top_k, len(target_flat))
    if effective_top_k_target == 0:
        print(f"Token Position {token_pos} Cannot retrieve target model probabilities top-k, Skip.")
        return

    target_topk_indices = np.argsort(target_flat)[-effective_top_k_target:][::-1]
    target_topk_probs_values = target_flat[target_topk_indices]

    draft_probs_for_target_tokens = draft_flat[target_topk_indices]

    effective_top_k_draft = min(top_k, len(draft_flat))
    if effective_top_k_draft == 0:
        print(f"Token Position {token_pos} Cannot retrieve draft model probabilities top-k, Skip.")
        return

    draft_topk_indices = np.argsort(draft_flat)[-effective_top_k_draft:][::-1]
    draft_topk_probs_values = draft_flat[draft_topk_indices]

    target_probs_for_draft_tokens = target_flat[draft_topk_indices]

    try:
        target_topk_tokens = [clean_token_text(tokenizer.decode([idx], skip_special_tokens=False), idx) for idx in target_topk_indices]
        draft_topk_tokens = [clean_token_text(tokenizer.decode([idx], skip_special_tokens=False), idx) for idx in draft_topk_indices]
    except Exception as e:
        print(f"Tokenizer Decoding failed: {e}. will only use Token ID.")
        target_topk_tokens = [clean_token_text(str(idx), idx) for idx in target_topk_indices]
        draft_topk_tokens = [clean_token_text(str(idx), idx) for idx in draft_topk_indices]

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    title = f'Token Position {token_pos} of Top-{top_k} Token Probability comparison' if use_chinese else f'Token Position {token_pos} - Top-{top_k} Token Probability Comparison'
    fig.suptitle(title, fontsize=18, y=0.98)

    ax1 = fig.add_subplot(gs[0, 0])
    title1 = "Target model Top-k Tokens" if use_chinese else 'Target Model Top-k Tokens'
    ax1.set_title(title1, fontsize=14)
    bars1 = ax1.bar(np.arange(effective_top_k_target), target_topk_probs_values, color='royalblue', alpha=0.8, width=0.8)
    ax1.set_xticks(np.arange(effective_top_k_target))
    ax1.set_xticklabels(target_topk_tokens, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel("Probability" if use_chinese else 'Probability', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    title2 = "Draft model versus target Top-k Tokens probabilities" if use_chinese else 'Draft Model for Target Top-k Tokens'
    ax2.set_title(title2, fontsize=14)
    bars2 = ax2.bar(np.arange(effective_top_k_target), draft_probs_for_target_tokens, color='mediumseagreen', alpha=0.8, width=0.8)
    ax2.set_xticks(np.arange(effective_top_k_target))
    ax2.set_xticklabels(target_topk_tokens, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel("Probability" if use_chinese else 'Probability', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax3 = fig.add_subplot(gs[1, 0])
    title3 = "Draft model Top-k Tokens" if use_chinese else 'Draft Model Top-k Tokens'
    ax3.set_title(title3, fontsize=14)
    bars3 = ax3.bar(np.arange(effective_top_k_draft), draft_topk_probs_values, color='mediumseagreen', alpha=0.8, width=0.8)
    ax3.set_xticks(np.arange(effective_top_k_draft))
    ax3.set_xticklabels(draft_topk_tokens, rotation=45, ha='right', fontsize=10)
    ax3.set_ylabel("Probability" if use_chinese else 'Probability', fontsize=12)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax4 = fig.add_subplot(gs[1, 1])
    title4 = "Target model versus draft Top-k Tokens probabilities" if use_chinese else 'Target Model for Draft Top-k Tokens'
    ax4.set_title(title4, fontsize=14)
    bars4 = ax4.bar(np.arange(effective_top_k_draft), target_probs_for_draft_tokens, color='royalblue', alpha=0.8, width=0.8)
    ax4.set_xticks(np.arange(effective_top_k_draft))
    ax4.set_xticklabels(draft_topk_tokens, rotation=45, ha='right', fontsize=10)
    ax4.set_ylabel("Probability" if use_chinese else 'Probability', fontsize=12)
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_dir = './logits_data/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/token_pos_{token_pos}_comparison.png', dpi=150)
    plt.close(fig)

    print(f"Saved Token Position {token_pos} visualization plots")

def compute_kl_divergence(p_probs, q_probs):
    """Compute KL divergence: KL(P||Q)."""
    if p_probs.size == 0 or q_probs.size == 0 or p_probs.shape != q_probs.shape:
        print("Probability distributions are empty or have mismatched shapes, Cannot compute KL divergence.")
        return np.nan

    p_probs = np.maximum(p_probs, 0)
    p_probs /= np.sum(p_probs)

    q_probs = np.maximum(q_probs, 0)
    q_probs_sum = np.sum(q_probs)
    if q_probs_sum == 0:
        print("Warning: q_probs all are 0,KL divergence is undefined or infinite.")
        return np.inf
    q_probs /= q_probs_sum

    epsilon = 1e-12

    kl_div = np.sum(np.where(p_probs > epsilon, p_probs * np.log(p_probs / np.maximum(q_probs, epsilon)), 0))
    return kl_div

def visualize_all_token_positions(token_positions, tokenizer_path, top_k=10):
    """Compare complete logit distributions at every token position."""
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print(f"Successfully loaded tokenizer, Vocabulary size: {tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)}")
    except Exception as e:
        print(f"Load tokenizer '{tokenizer_path}' failed: {e}")
        print("will use a simple token ID as labels")
        class DummyTokenizer:
            def decode(self, token_ids, skip_special_tokens=False):

                if not isinstance(token_ids, list) or not token_ids: return "ERR_ID"
                return f"ID:{token_ids[0]}"
        tokenizer = DummyTokenizer()

    kl_divergences = []

    for pos in token_positions:
        target_logits_file = f'./logits_data/target/logits_{pos}.bin'
        draft_logits_file = f'./logits_data/draft/logits_{pos}.bin'

        target_logits = read_tensor_from_file(target_logits_file)
        draft_logits = read_tensor_from_file(draft_logits_file)

        if target_logits is None or draft_logits is None:
            print(f"Skip token Position {pos}, because the file is missing or could not be read")
            continue

        if target_logits.ndim > 1: target_logits = target_logits.squeeze()
        if draft_logits.ndim > 1: draft_logits = draft_logits.squeeze()

        if target_logits.ndim != 1 or draft_logits.ndim != 1:
            print(f"Token Position {pos} of logits is not one-dimensional, Skip.Target shape: {target_logits.shape}, Draft shape: {draft_logits.shape}")
            continue

        target_probs = apply_softmax(target_logits)
        draft_probs = apply_softmax(draft_logits)

        if target_probs.size == 0 or draft_probs.size == 0:
            print(f"Token Position {pos} computed probabilities are empty, Skip.")
            continue

        visualize_top_tokens(target_probs, draft_probs, tokenizer, top_k, pos)

        kl_div = compute_kl_divergence(target_probs, draft_probs)
        if not np.isnan(kl_div) and not np.isinf(kl_div):
             kl_divergences.append((pos, kl_div))
        print(f"Token Position {pos} of KL divergence: {kl_div:.4f}")

    if kl_divergences:
        positions, divergences = zip(*kl_divergences)

        plt.figure(figsize=(12, 7))
        plt.bar(positions, divergences, color='mediumpurple', alpha=0.8, width=0.8)
        title_kl = "each Token at position KL divergence ( Target vs Draft )" if use_chinese else 'KL Divergence by Token Position (Target vs Draft)'
        plt.title(title_kl, fontsize=16)
        plt.xlabel("Token Position" if use_chinese else 'Token Position', fontsize=12)
        plt.ylabel('KL divergence (KL(Target || Draft))' if use_chinese else 'KL Divergence (KL(Target || Draft))', fontsize=12)
        plt.xticks(positions)
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()

        output_dir = './logits_data/visualizations'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/kl_divergence_comparison.png', dpi=150)
        plt.close()

        print(f"Saved KL divergence plot to {output_dir}/kl_divergence_comparison.png")

def main():
    target_dir = './logits_data/target'
    draft_dir = './logits_data/draft'

    if not os.path.exists(target_dir) or not os.path.isdir(target_dir) or \
       not os.path.exists(draft_dir) or not os.path.isdir(draft_dir):
        print(f"Error: logits Data directory '{target_dir}' or '{draft_dir}' does not exist or is not a directory. Run the script first to generate data.")
        return

    target_files = [f for f in os.listdir(target_dir) if f.startswith('logits_') and f.endswith('.bin')]
    token_positions = []
    for f_name in target_files:
        try:

            match = re.search(r'logits_(\d+)\.bin', f_name)
            if match:
                token_positions.append(int(match.group(1)))
        except ValueError:
            print(f"Warning: Cannot parse the file name {f_name} parse from token Position.")

    if not token_positions:
        print(f"Error: in '{target_dir}' contains no valid logits data files ( for example logits_0.bin). Run the script first to generate data.")
        return

    token_positions.sort()
    print(f"Found {len(token_positions)} items token at position logits data: {token_positions}")

    # tokenizer_path = "./models/Qwen3-1.7B-AWQ"

    # tokenizer_path = "bert-base-multilingual-cased"

    tokenizer_path = "./models/Qwen3-1.7B-AWQ"
    print(f"will try to load from '{tokenizer_path}' Load tokenizer. if unsuccessful, will use Token ID as labels.")
    print("Ensure this path contains valid tokenizer Model file ( for example tokenizer.json, vocab.txt/json, spiece.model etc. ).")

    visualize_all_token_positions(token_positions, tokenizer_path)

if __name__ == "__main__":

    print("Starting the visualization script...")
    print(f"Current Matplotlib Font configuration (font.family): {plt.rcParams['font.family']}")
    print(f"Current Matplotlib Font configuration (font.sans-serif): {plt.rcParams['font.sans-serif']}")
    main()
    print("Visualization script complete.")