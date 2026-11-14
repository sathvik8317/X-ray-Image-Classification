"""CNN vs. MedGemma: is disagreement between the two a useful uncertainty signal?

Two very different models look at every test X-ray:
  * the project's EfficientNetV2-B0 (pneumonia probabilities in --cnn-probs, from
    `cnn_probs.py`), and
  * MedGemma 4B, a medical vision-language model, asked a yes/no question. Its pneumonia
    probability is P("yes") / (P("yes") + P("no")) for the first answer token, read
    straight from the logits, so there is no text parsing and no sampling.

Then: how good is MedGemma zero-shot (AUC)? How often do the two agree? Is the CNN much
less accurate on the images where they disagree? If so, disagreement can flag cases for a
human, with no labels and no retraining. For the most confident disagreements MedGemma
also writes a short findings paragraph, to read next to the Grad-CAM.

Needs torch, transformers and accelerate (a Kaggle GPU image has them) and the MedGemma
weights, which require accepting Google's Health AI Developer Foundations terms.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForImageTextToText, AutoProcessor

QUESTION = (
    "This is a pediatric chest X-ray. Is there radiographic evidence of pneumonia? "
    "Answer with one word: yes or no."
)
FINDINGS = (
    "You are a pediatric radiologist. Describe the findings in this chest X-ray in two or "
    "three sentences, then give an impression: pneumonia or no pneumonia."
)


def answer_token_ids(processor, words: tuple[str, ...]) -> list[int]:
    """First-token ids of each spelling of the answer words (yes, Yes, " yes", ...)."""
    ids = set()
    for word in words:
        for variant in (word, word.capitalize(), " " + word, " " + word.capitalize()):
            ids.add(processor.tokenizer.encode(variant, add_special_tokens=False)[0])
    return sorted(ids)


def messages(image: Image.Image, text: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": text}],
        }
    ]


def inputs_for(processor, model, image: Image.Image, text: str):
    return processor.apply_chat_template(
        messages(image, text),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)


@torch.inference_mode()
def p_yes(model, processor, image: Image.Image, yes: list[int], no: list[int]) -> float:
    logits = model(**inputs_for(processor, model, image, QUESTION)).logits[0, -1].float()
    log_yes = torch.logsumexp(logits[yes], 0)
    log_no = torch.logsumexp(logits[no], 0)
    return torch.sigmoid(log_yes - log_no).item()


@torch.inference_mode()
def findings(model, processor, image: Image.Image) -> str:
    inputs = inputs_for(processor, model, image, FINDINGS)
    out = model.generate(**inputs, max_new_tokens=160, do_sample=False)
    return processor.decode(out[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)


def summarize(y: np.ndarray, cnn: np.ndarray, vlm: np.ndarray, cnn_t: float) -> dict:
    cnn_pred, vlm_pred = cnn >= cnn_t, vlm >= 0.5
    agree = cnn_pred == vlm_pred
    correct = cnn_pred == y.astype(bool)

    def acc(mask):
        return float(correct[mask].mean()) if mask.any() else float("nan")

    return {
        "n": len(y),
        "auc_cnn": float(roc_auc_score(y, cnn)),
        "auc_medgemma": float(roc_auc_score(y, vlm)),
        "auc_average": float(roc_auc_score(y, (cnn + vlm) / 2)),
        "medgemma_sensitivity": float(vlm_pred[y == 1].mean()),
        "medgemma_specificity": float((~vlm_pred)[y == 0].mean()),
        "agreement_rate": float(agree.mean()),
        "cnn_accuracy_when_agree": acc(agree),
        "cnn_accuracy_when_disagree": acc(~agree),
        "share_of_cnn_errors_flagged": float((~agree & ~correct).sum() / max((~correct).sum(), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--cnn-probs",
        type=Path,
        required=True,
        help="JSON from cnn_probs.py: paths, labels, probs, threshold",
    )
    parser.add_argument("--out", type=Path, default=Path("medgemma_results.json"))
    parser.add_argument("--n-findings", type=int, default=8)
    args = parser.parse_args()

    cnn = json.loads(args.cnn_probs.read_text())
    processor = AutoProcessor.from_pretrained(args.model_dir)
    # T4s have no bfloat16 and Gemma 3 overflows in float16, so use float32 over both GPUs.
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, dtype=torch.float32, device_map="auto"
    ).eval()
    yes, no = answer_token_ids(processor, ("yes",)), answer_token_ids(processor, ("no",))

    probs = []
    for i, path in enumerate(cnn["paths"]):
        probs.append(p_yes(model, processor, Image.open(path).convert("RGB"), yes, no))
        if i % 50 == 0:
            print(f"{i}/{len(cnn['paths'])} p_yes={probs[-1]:.3f}", flush=True)

    y, cnn_p, vlm_p = np.array(cnn["labels"]), np.array(cnn["probs"]), np.array(probs)
    summary = summarize(y, cnn_p, vlm_p, cnn["threshold"])
    print(json.dumps(summary, indent=2))

    # Most confident disagreements: both models sure, opposite answers.
    disagreement = np.abs(cnn_p - vlm_p) * ((cnn_p >= cnn["threshold"]) != (vlm_p >= 0.5))
    cases = []
    for i in np.argsort(-disagreement)[: args.n_findings]:
        text = findings(model, processor, Image.open(cnn["paths"][i]).convert("RGB"))
        cases.append(
            {
                "file": Path(cnn["paths"][i]).name,
                "label": int(y[i]),
                "cnn": float(cnn_p[i]),
                "medgemma": float(vlm_p[i]),
                "findings": text,
            }
        )
        print(
            f"\n{cases[-1]['file']} label={y[i]} cnn={cnn_p[i]:.2f} medgemma={vlm_p[i]:.2f}\n{text}"
        )

    args.out.write_text(
        json.dumps(
            {
                "summary": summary,
                "medgemma_probs": probs,
                "files": [Path(p).name for p in cnn["paths"]],
                "disagreements": cases,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
