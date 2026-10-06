# Gradio app: loads the fine-tuned model from the HF Hub and classifies pasted articles.
import os
import sys
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kinnews.text import normalize  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "Samkwizera/kinnews-topic-classifier")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 512))
LOW_CONFIDENCE = 0.5

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

KIN_NAMES = {
    "politics": "Politiki", "sport": "Imikino", "economy": "Ubukungu", "health": "Ubuzima",
    "entertainment": "Imyidagaduro", "history": "Amateka", "technology": "Ikoranabuhanga",
    "tourism": "Ubukerarugendo", "culture": "Umuco", "fashion": "Imideri", "religion": "Iyobokamana",
    "environment": "Ibidukikije", "education": "Uburezi", "relationship": "Urukundo",
}


@torch.no_grad()
def classify(title: str, body: str):
    # same normalisation as training, otherwise apostrophes etc. tokenise differently
    text = normalize(f"{title}. {body}" if title.strip() else body)
    if len(text.split()) < 3:
        return {}, "Please enter at least a sentence of Kinyarwanda text."
    enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    probs = torch.softmax(model(**enc).logits[0], dim=-1)
    scores = {f"{label} ({KIN_NAMES[label]})": float(probs[i]) for i, label in model.config.id2label.items()}

    n_tokens = len(tokenizer(text)["input_ids"])
    notes = [f"Input: {n_tokens} subword tokens" + (f" (only the first {MAX_LENGTH} were used)" if n_tokens > MAX_LENGTH else "")]
    if probs.max() < LOW_CONFIDENCE:
        notes.append(f"Low confidence ({probs.max():.0%}): the article may cover several topics, or a topic "
                     "that was rare in the training data.")
    return scores, "\n\n".join(notes)


EXAMPLES = [
    ["Amavubi yatsinze Benin ibitego 2-0", "Ikipe y'igihugu y'umupira w'amaguru, Amavubi, yatsinze ikipe ya Benin "
     "ibitego bibiri ku busa mu mukino wo gushaka itike y'igikombe cy'Afurika wabereye kuri Stade Amahoro i Kigali."],
    ["Abahinzi barasabwa gukoresha ifumbire", "Minisiteri y'Ubuhinzi n'Ubworozi irasaba abahinzi gukoresha ifumbire "
     "y'imborera n'imvaruganda kugira ngo umusaruro w'ibigori n'ibishyimbo wiyongere muri iki gihembwe cy'ihinga."],
    ["Abarwayi ba malariya bagabanutse", "Ikigo cy'igihugu cy'ubuzima (RBC) cyatangaje ko umubare w'abarwara "
     "malariya wagabanutse ku kigero cya 30% bitewe no gukoresha inzitiramibu no gutera imiti mu mazu."],
]

with gr.Blocks(title="KINNEWS Topic Classifier") as demo:
    gr.Markdown(
        "# Kinyarwanda News Topic Classifier\n"
        "Paste a Kinyarwanda news article and the model predicts its topic among 14 categories. "
        f"Model: [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}), fine-tuned on the KINNEWS corpus."
    )
    with gr.Row():
        with gr.Column():
            title = gr.Textbox(label="Title (Umutwe)", lines=1)
            body = gr.Textbox(label="Article text (Inkuru)", lines=10)
            btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            out = gr.Label(num_top_classes=5, label="Predicted topic")
            info = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=[title, body])
    btn.click(classify, inputs=[title, body], outputs=[out, info])

if __name__ == "__main__":
    demo.launch()
