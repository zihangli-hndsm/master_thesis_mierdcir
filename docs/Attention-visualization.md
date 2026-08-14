# Role
You are an expert Computer Vision and NLP Engineer proficient in PyTorch, CLIP, OpenCV, and matplotlib.

# Task
Write a standalone Python diagnostic script (`visualize_attention.py`) for qualitative analysis in a Composed Image Retrieval (CIR) thesis. Given a sample ID from the annotated dataset, the script must load the trained model checkpoint, extract token-level spatial cross-attention maps for specific noun phrases (NPs), and generate a high-quality visualization figure for publication.

# Technical Background & Architecture (ScheiCIR)
- **[span_0](start_span)Model Backbone:** CLIP (ViT-B)[span_0](end_span).
- **[span_1](start_span)Core Mechanism:** Noun phrases $\mathcal{N}=\{n_1, n_2, ..., n_K\}$ are extracted from the text[span_1](end_span). [span_2](start_span)An attention module computes NP-specific spatial attention maps $A_i = Attention(t_i, F)$ where $t_i$ is the text embedding of the $i$-th NP, and $F$ denotes the reference visual feature maps[span_2](end_span).
- **[span_3](start_span)Thesis Diagnosis:** The soft contrastive loss over NP pairs saturates early because the positive term uses self-similarity $sim(A_i, A_i)$[span_3](end_span). We need this visualization tool to empirically prove that attention maps under different NPs are degenerated/overlapping in Fail Cases.

# Functional Requirements

## 1. Input & Data Loading
- The script should accept a command-line argument `--id [SAMPLE_ID]` or a simple prompt input.
- Load `samples.json` to fetch the corresponding paths: `reference_path`, `target_path`, `mtcir_text`, `merdcir_text`.
- [span_4](start_span)Load the trained PyTorch checkpoint (`checkpoints/topk*/... .pth.tar`)[span_4](end_span).

## 2. Attention Extraction Pipeline
- Implement a forward hooks or direct function call to extract:
  1. The spatial feature maps $F$ from the last layer of the CLIP Visual Encoder before global pooling.
  2. The text tokens/embeddings $t_i$ for each noun phrase parsed from the text.
  3. [span_5](start_span)The raw spatial attention matrix $A_i$ before any flattening or pooling[span_5](end_span).

## 3. Heatmap Generation & Overlay Logic (OpenCV/Matplotlib)
- Rescale the low-resolution attention map $A_i$ to match the original `Reference Image` dimensions using bilinear interpolation (`cv2.resize`).
- Apply a jet colormap (`cv2.applyColorMap(..., cv2.COLORMAP_JET)`) to convert the normalized attention weights into a pseudo-color heatmap.
- Overlay the heatmap onto the original reference image with a blending weight (e.g., $\alpha=0.6$ for image, $\beta=0.4$ for heatmap) using `cv2.addWeighted`.

## 4. Multi-Panel Visualization Layout (for Paper Submission)
The script must use `matplotlib.pyplot` to generate a single, clean figure with a multi-panel layout:
- **Panel 1:** Original Reference Image (with Text titles).
- **[span_6](start_span)Panel 2:** Original Target Image (the true retrieval target)[span_6](end_span).
- **Panel 3 to N:** The generated Heatmap overlays. Create one distinct subplot for *each* extracted Noun Phrase (NP) in the text. Explicitly title each subplot with its corresponding noun phrase string (e.g., "Attention for: 'wooden bookshelf'").
- **Styling:** Remove axes (`plt.axis('off')`), use tight layout (`plt.tight_layout()`), and save the final figure directly as a publication-ready vector graphic format: `plots/attention_fail_case_[ID].pdf`.

# Output Specification
Provide a robust, modular Python script. Use placeholders or mock modules for the specific `ScheiCIR` model class architecture, but implement the full OpenCV heatmap overlay and Matplotlib plotting pipeline logically so it can be adapted immediately.