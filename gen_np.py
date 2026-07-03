import multiprocessing

# Set the multiprocessing start method before any parallel or CUDA work.
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import argparse
import contextlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import spacy
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from transformers import CLIPTokenizerFast
import torch
import re
import unicodedata

def clean_text(text: str) -> str:
    if not text:
        return ""
    # 1. Normalize Unicode so equivalent characters share one encoding.
    text = unicodedata.normalize("NFKC", text)
    # 2. Remove invisible/control characters while preserving newlines and tabs.
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    # 3. Remove zero-width and other problematic formatting characters.
    text = re.sub(r'[\u200b\u200e\u200f\ufeff\xad]', '', text)
    # 4. Collapse repeated whitespace to a single space.
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _install_transformers_compat_shims() -> None:
    """
    Backward-compatibility helpers for libraries (e.g. benepar) that still call
    tokenizer methods removed in newer transformers versions.
    """
    if not hasattr(PreTrainedTokenizerBase, "_eventually_correct_t5_max_length"):
        def _eventually_correct_t5_max_length(self, pretrained_model_name_or_path, max_model_length, init_max_model_length):
            return max_model_length

        PreTrainedTokenizerBase._eventually_correct_t5_max_length = _eventually_correct_t5_max_length  # type: ignore[attr-defined]

    if not hasattr(PreTrainedTokenizerBase, "as_target_tokenizer"):
        @contextlib.contextmanager
        def as_target_tokenizer(self):
            yield self

        PreTrainedTokenizerBase.as_target_tokenizer = as_target_tokenizer  # type: ignore[attr-defined]


_install_transformers_compat_shims()

import benepar


STOPWORDS = {"Change"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract leaf noun phrases (NPs) with benepar + spaCy and map them to "
            "CLIP token spans. Supports resume from the last written id."
        )
    )
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL path.")
    parser.add_argument("--output-jsonl", default="output_nps.jsonl", help="Output JSONL path.")
    parser.add_argument("--spacy-model", default="en_core_web_md", help="spaCy model name.")
    parser.add_argument("--benepar-model", default="benepar_en3", help="benepar model name.")
    parser.add_argument("--batch-size", type=int, default=2048, help="spaCy pipe batch size.")
    parser.add_argument("--max-length", type=int, default=50000, help="spaCy max_length.")
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="spaCy n_process when running on CPU.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last id already present in output JSONL.",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def last_processed_id(output_path: str) -> Optional[Any]:
    if not os.path.exists(output_path):
        return None

    last_obj: Optional[Dict[str, Any]] = None
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last_obj = json.loads(line)
            except json.JSONDecodeError:
                continue

    if not last_obj:
        return None
    return last_obj.get("id")


def skip_to_resume_position(records: Sequence[Dict[str, Any]], resume_id: Any) -> Tuple[int, Sequence[Dict[str, Any]]]:
    for i, rec in enumerate(records):
        if rec.get("id") == resume_id:
            return i + 1, records[i + 1 :]
    return 0, records


def get_text(record: Dict[str, Any]) -> Optional[str]:
    # Resolve the modification text from the supported input fields.
    raw_text = None
    text_source = record.get("merdcir_modification")
    if isinstance(text_source, str) and text_source.strip():
        raw_text = text_source
    else:
        modification = record.get("modification")
        if isinstance(modification, list):
            sentences = [str(x).strip(". ") for x in modification if str(x).strip()]
            if sentences:
                raw_text = ". ".join(sentences)
        elif isinstance(modification, str) and modification.strip():
            raw_text = modification

    # Clean resolved text before parsing.
    if raw_text:
        cleaned = clean_text(raw_text)
        return cleaned if cleaned else None
    return None


def extract_leaf_nps(doc, hf_tokenizer: CLIPTokenizerFast) -> List[List[Any]]:
    leaf_nps: List[List[Any]] = []
    inputs = hf_tokenizer(doc.text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = inputs["offset_mapping"]

    for sent in doc.sents:
        def find_leaf_nps(node) -> None:
            if "NP" in node._.labels:
                has_child_np = any(
                    "NP" in child._.labels for child in node._.children if hasattr(child._, "labels")
                )
                if not has_child_np:
                    token_indices: List[int] = []
                    for i, (ts, te) in enumerate(offsets):
                        if ts == te == 0:
                            continue
                        if ts >= node.start_char and te <= node.end_char:
                            token_indices.append(i)

                    if token_indices and node.text not in STOPWORDS:
                        leaf_nps.append([node.text, [token_indices[0], token_indices[-1]]])
                    return

            for child in node._.children:
                if hasattr(child._, "labels"):
                    find_leaf_nps(child)

        find_leaf_nps(sent)

    return leaf_nps


def build_nlp(spacy_model: str, benepar_model: str, max_length: int):
    import torch
    import spacy

    # 1. Prefer GPU execution before loading spaCy/benepar models.
    activated = spacy.prefer_gpu()

    # 2. Load the base spaCy model.
    nlp = spacy.load(spacy_model, disable=["ner", "lemmatizer", "attribute_ruler"])

    # 3. Attach the benepar constituency parser.
    nlp.add_pipe("benepar", config={"model": benepar_model})

    # 4. Move benepar internals to GPU when the installed version exposes them.
    if "benepar" in nlp.pipe_names:
        component = nlp.get_pipe("benepar")
        parser = getattr(component, "_parser", None)

        if parser:
            # Check both known parser model attributes used by benepar versions.
            model = getattr(parser, "model", getattr(parser, "_model", None))

            if model is not None and isinstance(model, torch.nn.Module):
                model.cuda()
                print(f"✅ Success: detected model and moved it to GPU: {next(model.parameters()).device}")
            elif isinstance(parser, torch.nn.Module):
                # Some benepar versions expose the parser itself as an nn.Module.
                parser.cuda()
                print("✅ Success: parser object is an nn.Module and was moved to GPU")
            else:
                print("⚠️ Warning: found parser but could not identify its internal torch model object")
        else:
            print("❌ Error: could not find _parser in the benepar component")

    nlp.max_length = max_length
    return nlp, activated


def generate(records: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    hf_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    nlp, use_gpu = build_nlp(args.spacy_model, args.benepar_model, args.max_length)

    # Build the cleaned text batch used by spaCy pipe.
    valid_data = []
    for rec in records:
        text = get_text(rec)
        if text:
            # Keep source metadata next to the cleaned text for output reconstruction.
            valid_data.append({"rec": rec, "text": text})

    texts = [d["text"] for d in valid_data]
    metadata = [d["rec"] for d in valid_data]

    # Use a single spaCy worker for benepar to keep GPU execution reproducible.
    pipe_iter = nlp.pipe(texts, batch_size=args.batch_size, n_process=1)

    for rec, doc in tqdm(zip(metadata, pipe_iter), total=len(metadata), desc="Extracting"):
        try:
            # Skip only the record that still fails parser assertions or token alignment.
            nps = extract_leaf_nps(doc, hf_tokenizer)
            if not nps:
                continue
            yield {
                "id": rec.get("id"),
                "image": rec.get("image"),
                "target_img": rec.get("target_image") or rec.get("target_img"),
                "modification": rec.get("merdcir_modification"),
                "nps": nps,
            }
        except Exception as e:
            # Log the failing record id and continue the extraction run.
            print(f"\n⚠️ Skipping ID {rec.get('id')}: encountered error {type(e).__name__}: {e}")
            continue


def main() -> None:
    args = parse_args()

    records = load_jsonl(args.input_jsonl)
    start_idx = 0

    if args.resume:
        resume_id = last_processed_id(args.output_jsonl)
        if resume_id is not None:
            start_idx, records = skip_to_resume_position(records, resume_id)
            print(f"[resume] last id={resume_id!r}, continuing from index {start_idx}.")
        else:
            print("[resume] no previous output found, starting from scratch.")

    write_mode = "a" if args.resume else "w"
    with open(args.output_jsonl, write_mode, encoding="utf-8") as out_f:
        for item in generate(records, args):
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
