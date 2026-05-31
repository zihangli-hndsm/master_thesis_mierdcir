# Role
You are an expert Full-Stack Python Developer specializing in PyTorch, Multimodal VLMs, and Gradio UI development.

# Task
Write a single-file Python script (`app.py`) using Gradio to build a local, standalone human evaluation platform for Composed Image Retrieval (CIR) text rewrite quality.

# Background & Requirement
The goal is to evaluate the rewrite quality of the MiERDCIR discipline against the original MTCIR text. 
The system runs locally on a Linux system and reads local absolute image paths. No port forwarding or external databases are required. All data is saved to a local CSV file.

## 1. Data Preparation & Splitting Logic
- The script should check for a local `samples.json` which contains a list of 160 sampled dataset items.
- Each item in `samples.json` must have: `id`, `reference_path`, `target_path`, `mtcir_text`, `merdcir_text`.
- **Annotator Assignment:** The script must take a command-line argument or an initial UI dropdown to select the Annotator Name (e.g., "Annotator_1" or "Annotator_2").
- **Data Division:** - If "Annotator_1" is selected, load samples 0 to 79 (First 80 examples).
  - If "Annotator_2" is selected, load samples 80 to 159 (Last 80 examples).
- For a rigorous blind test, the UI must randomly swap the display positions of `mtcir_text` and `merdcir_text` (Label them as "Text Option A" and "Text Option B" in the UI). The script must internally track which one is which when saving the results.

## 2. UI Layout Requirements (Gradio)
- **Top Row (Images Side-by-Side):** Display `Reference Image` and `Target Image` using `gr.Image` components with their absolute local Linux paths.
- **Middle Row (Text Comparison Layout):**
  - Display "Text Option A (MTCIR/MiERDCIR)" and "Text Option B (MTCIR/MiERDCIR)" separated or connected visually by an arrow symbol (`->`) to represent the modification rewriting trajectory.
- **Right/Bottom Panel (Evaluation Sliders & Checkboxes):**
  - Provide 4 Slider bars (`gr.Slider`) with a scale of **0 to 100** for the following metrics:
    1. **Intent Preservation** (Does the rewrite keep the original core editing goal?)
    2. **Naturalness / User-likeness** (Does it sound like a real human search query?)
    3. **Discriminativeness for Retrieval** (Is it clear enough for a retriever to find the target?)
    4. **Harmful Omission or Hallucination** (Did it accidentally drop critical attributes or invent fake details?)
  - Provide a single Checkbox (`gr.Checkbox`): **"Mark as a typical Fail Case for MiERDCIR"**.
  - Provide a notes text area (`gr.Textbox`) for qualitative comments.

## 3. Data Saving & Control Flow
- **Submit Button:** Clicking "Submit Annotation" must:
  1. Map the scores of "Text Option A/B" back to their true identities (`mtcir_text` or `merdcir_text`).
  2. Append a new row into a local file named `evaluation_results_[annotator_name].csv`.
  3. The CSV columns must include: `sample_id`, `annotator`, `mtcir_intent_preservation`, `merdcir_intent_preservation`, `mtcir_naturalness`, `merdcir_naturalness`, ..., `is_mierdcir_fail_case`, `notes`, `timestamp`.
  4. Automatically clear the inputs, refresh the random text swapping, and load the next sample image triplet.
- Progress tracking: Display an explicit progress text (e.g., "Current Progress: 14 / 80") so the annotator knows their status.

# Output Specification
Generate a single, clean, robust `app.py` file. Include a mockup `samples.json` generator function at the beginning of the file that creates dummy paths if the file doesn't exist, ensuring the script can be tested instantly.