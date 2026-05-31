#!/usr/bin/env python3
"""Local Gradio app for human evaluation of CIR rewrite quality."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import gradio as gr


SAMPLE_COUNT = 160
SPLIT_SIZE = 80
ANNOTATORS = ("Annotator_1", "Annotator_2")
SAMPLE_COLUMNS = ("id", "reference_path", "target_path", "mtcir_text", "merdcir_text")
METRICS = (
    ("intent_preservation", "Intent Preservation"),
    ("naturalness", "Naturalness / User-likeness"),
    ("discriminativeness", "Discriminativeness for Retrieval"),
    ("harmful_omission_or_hallucination", "Harmful Omission or Hallucination"),
)
CSV_COLUMNS = [
    "sample_id",
    "annotator",
    "mtcir_text",
    "merdcir_text",
    "mtcir_intent_preservation",
    "merdcir_intent_preservation",
    "mtcir_naturalness",
    "merdcir_naturalness",
    "mtcir_discriminativeness",
    "merdcir_discriminativeness",
    "mtcir_harmful_omission_or_hallucination",
    "merdcir_harmful_omission_or_hallucination",
    "is_mierdcir_fail_case",
    "notes",
    "timestamp",
]


def create_mock_samples_if_missing(samples_path: Path) -> None:
    """Create a self-contained mock sample file so the app can run immediately."""
    if samples_path.exists():
        return

    mock_dir = samples_path.parent / "mock_images"
    mock_dir.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image, ImageDraw

        for name, color, label in (
            ("reference.jpg", (226, 232, 240), "Reference"),
            ("target.jpg", (207, 250, 229), "Target"),
        ):
            image_path = mock_dir / name
            image = Image.new("RGB", (640, 420), color)
            draw = ImageDraw.Draw(image)
            draw.rectangle((28, 28, 612, 392), outline=(45, 55, 72), width=4)
            draw.text((250, 190), label, fill=(17, 24, 39))
            image.save(image_path)
    except Exception:
        # Gradio will still start and show missing-image warnings in the UI.
        pass

    reference_path = str(Path("mock_images") / "reference.jpg")
    target_path = str(Path("mock_images") / "target.jpg")
    samples = []
    for idx in range(SAMPLE_COUNT):
        samples.append(
            {
                "id": f"mock_{idx:03d}",
                "reference_path": reference_path,
                "target_path": target_path,
                "mtcir_text": "Change the product photo to a brighter studio shot with a plain background.",
                "merdcir_text": (
                    "Make the image look like a clean studio product photo with brighter lighting "
                    "and a simple plain background."
                ),
            }
        )

    with samples_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)


def load_samples(samples_path: Path) -> List[Dict[str, Any]]:
    create_mock_samples_if_missing(samples_path)
    with samples_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        raise ValueError(f"{samples_path} must contain a JSON list.")
    if len(samples) < SAMPLE_COUNT:
        raise ValueError(f"{samples_path} must contain at least {SAMPLE_COUNT} samples.")

    for idx, sample in enumerate(samples[:SAMPLE_COUNT]):
        missing = [key for key in SAMPLE_COLUMNS if key not in sample]
        if missing:
            raise ValueError(f"Sample {idx} is missing required keys: {', '.join(missing)}")
    return samples[:SAMPLE_COUNT]


def result_path_for(annotator: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in annotator)
    return Path(f"evaluation_results_{safe_name}.csv")


def completed_sample_ids(annotator: str) -> Set[str]:
    path = result_path_for(annotator)
    if not path.exists():
        return set()

    with path.open("r", newline="", encoding="utf-8") as f:
        return {row["sample_id"] for row in csv.DictReader(f) if row.get("sample_id")}


def annotator_slice(annotator: str) -> Tuple[int, int]:
    if annotator == "Annotator_1":
        return 0, SPLIT_SIZE
    if annotator == "Annotator_2":
        return SPLIT_SIZE, SAMPLE_COUNT
    raise ValueError(f"Unknown annotator: {annotator}")


def split_samples(samples: Sequence[Dict[str, Any]], annotator: str) -> List[Dict[str, Any]]:
    start, end = annotator_slice(annotator)
    return list(samples[start:end])


def first_unfinished_index(samples: Sequence[Dict[str, Any]], annotator: str) -> int:
    done = completed_sample_ids(annotator)
    for idx, sample in enumerate(samples):
        if str(sample["id"]) not in done:
            return idx
    return len(samples)


def existing_image(path_value: Any) -> Optional[str]:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path) if path.is_file() else None


def image_warning(sample: Dict[str, Any]) -> str:
    missing = []
    for label, key in (("Reference", "reference_path"), ("Target", "target_path")):
        if existing_image(sample.get(key)) is None:
            missing.append(f"{label}: {sample.get(key)}")
    if not missing:
        return ""
    return "Missing local image file(s):\n" + "\n".join(missing)


def empty_outputs(message: str) -> Tuple[Any, ...]:
    return (
        None,
        None,
        "",
        "",
        message,
        {"index": 0, "sample_id": None},
    )


def render_sample(annotator: str, index: Optional[int] = None) -> Tuple[Any, ...]:
    if annotator not in ANNOTATORS:
        return empty_outputs("Select an annotator to begin.")

    assigned = split_samples(SAMPLES, annotator)
    sample_index = first_unfinished_index(assigned, annotator) if index is None else int(index)
    if sample_index >= len(assigned):
        return empty_outputs(f"Current Progress: {len(assigned)} / {len(assigned)}. All assigned samples are complete.")

    sample = assigned[sample_index]
    progress = f"Current Progress: {sample_index + 1} / {len(assigned)}"
    warning = image_warning(sample)
    if warning:
        progress = f"{progress}\n\n{warning}"

    state = {
        "index": sample_index,
        "sample_id": str(sample["id"]),
    }
    return (
        existing_image(sample["reference_path"]),
        existing_image(sample["target_path"]),
        str(sample["mtcir_text"]),
        str(sample["merdcir_text"]),
        progress,
        state,
    )


def append_result(row: Dict[str, Any], annotator: str) -> None:
    path = result_path_for(annotator)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def submit_annotation(
    annotator: str,
    state: Dict[str, Any],
    mtcir_intent: float,
    merdcir_intent: float,
    mtcir_naturalness: float,
    merdcir_naturalness: float,
    mtcir_discriminativeness: float,
    merdcir_discriminativeness: float,
    mtcir_harmful: float,
    merdcir_harmful: float,
    is_fail_case: bool,
    notes: str,
) -> Tuple[Any, ...]:
    if annotator not in ANNOTATORS:
        return (*empty_outputs("Select an annotator before submitting."), 50, 50, 50, 50, 50, 50, 0, 0, False, "")
    if not state or state.get("sample_id") is None:
        return (*empty_outputs("No active sample to submit."), 50, 50, 50, 50, 50, 50, 0, 0, False, "")

    assigned = split_samples(SAMPLES, annotator)
    sample = next((item for item in assigned if str(item["id"]) == str(state["sample_id"])), None)
    if sample is None:
        return (*empty_outputs("Active sample was not found in the selected annotator split."), 50, 50, 50, 50, 50, 50, 0, 0, False, "")

    append_result(
        {
            "sample_id": state["sample_id"],
            "annotator": annotator,
            "mtcir_text": str(sample["mtcir_text"]),
            "merdcir_text": str(sample["merdcir_text"]),
            "mtcir_intent_preservation": mtcir_intent,
            "merdcir_intent_preservation": merdcir_intent,
            "mtcir_naturalness": mtcir_naturalness,
            "merdcir_naturalness": merdcir_naturalness,
            "mtcir_discriminativeness": mtcir_discriminativeness,
            "merdcir_discriminativeness": merdcir_discriminativeness,
            "mtcir_harmful_omission_or_hallucination": mtcir_harmful,
            "merdcir_harmful_omission_or_hallucination": merdcir_harmful,
            "is_mierdcir_fail_case": bool(is_fail_case),
            "notes": notes or "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        annotator,
    )

    rendered = render_sample(annotator)
    return (*rendered, 50, 50, 50, 50, 50, 50, 0, 0, False, "")


def build_app(default_annotator: Optional[str]) -> gr.Blocks:
    with gr.Blocks(title="CIR Rewrite Human Evaluation") as demo:
        gr.Markdown("# CIR Rewrite Human Evaluation")
        state = gr.State({})

        with gr.Row():
            annotator = gr.Dropdown(
                choices=list(ANNOTATORS),
                value=default_annotator,
                label="Annotator",
                interactive=default_annotator is None,
            )
            progress = gr.Textbox(label="Progress", lines=3, interactive=False)

        with gr.Row():
            reference_image = gr.Image(label="Reference Image", type="filepath", height=360)
            target_image = gr.Image(label="Target Image", type="filepath", height=360)

        with gr.Row(equal_height=True):
            option_a_text = gr.Textbox(
                label="Original Text (MTCIR)",
                lines=6,
                interactive=False,
            )
            gr.Markdown("## ->")
            option_b_text = gr.Textbox(
                label="Rewritten Text (MiERDCIR)",
                lines=6,
                interactive=False,
            )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Original MTCIR Scores")
                mtcir_intent = gr.Slider(0, 100, value=50, step=1, label=METRICS[0][1])
                mtcir_naturalness = gr.Slider(0, 100, value=50, step=1, label=METRICS[1][1])
                mtcir_discriminativeness = gr.Slider(0, 100, value=50, step=1, label=METRICS[2][1])
                mtcir_harmful = gr.Slider(0, 100, value=0, step=1, label=METRICS[3][1])
            with gr.Column():
                gr.Markdown("### Rewritten MiERDCIR Scores")
                merdcir_intent = gr.Slider(0, 100, value=50, step=1, label=METRICS[0][1])
                merdcir_naturalness = gr.Slider(0, 100, value=50, step=1, label=METRICS[1][1])
                merdcir_discriminativeness = gr.Slider(0, 100, value=50, step=1, label=METRICS[2][1])
                merdcir_harmful = gr.Slider(0, 100, value=0, step=1, label=METRICS[3][1])

        is_fail_case = gr.Checkbox(label="Mark as a typical Fail Case for MiERDCIR", value=False)
        notes = gr.Textbox(label="Notes", lines=4, placeholder="Optional qualitative comments")
        submit = gr.Button("Submit Annotation", variant="primary")

        sample_outputs = [
            reference_image,
            target_image,
            option_a_text,
            option_b_text,
            progress,
            state,
        ]
        reset_outputs = [
            mtcir_intent,
            merdcir_intent,
            mtcir_naturalness,
            merdcir_naturalness,
            mtcir_discriminativeness,
            merdcir_discriminativeness,
            mtcir_harmful,
            merdcir_harmful,
            is_fail_case,
            notes,
        ]
        score_inputs = [
            mtcir_intent,
            merdcir_intent,
            mtcir_naturalness,
            merdcir_naturalness,
            mtcir_discriminativeness,
            merdcir_discriminativeness,
            mtcir_harmful,
            merdcir_harmful,
            is_fail_case,
            notes,
        ]

        demo.load(render_sample, inputs=[annotator], outputs=sample_outputs)
        annotator.change(render_sample, inputs=[annotator], outputs=sample_outputs)
        submit.click(
            submit_annotation,
            inputs=[annotator, state, *score_inputs],
            outputs=[*sample_outputs, *reset_outputs],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CIR rewrite human evaluation app.")
    parser.add_argument("--samples", default="samples.json", help="Path to samples.json.")
    parser.add_argument("--annotator", choices=ANNOTATORS, help="Annotator split to load.")
    parser.add_argument("--server-name", default="127.0.0.1", help="Gradio server host.")
    parser.add_argument("--server-port", type=int, default=7860, help="Gradio server port.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link.")
    return parser.parse_args()


args = parse_args()
SAMPLES = load_samples(Path(args.samples))
app = build_app(args.annotator)


if __name__ == "__main__":
    app.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)
