# ============================================================
# "Real or AI?" game
# Human vs fine-tuned MobileNetV3Small
# Author: Ayush Varma
# Run locally:  python app.py   then open http://localhost:7860
# ============================================================
import os
import random

import numpy as np
import gradio as gr
import keras
from PIL import Image

AUTHOR = "Ayush Varma"
MODEL_PATH = "best_model.keras"
IMAGE_DIR = "test_images"      # contains real/ and ai/ subfolders
N_ROUNDS = 10
LABEL = {0: "AI-generated", 1: "Real"}

# ---- 1. Load the model once, when the app starts ----
model = keras.models.load_model(MODEL_PATH)


# ---- 2. Build the list of (image path, true label) ----
def load_image_list():
    items = []
    for folder_name, label in [("real", 1), ("ai", 0)]:
        folder = os.path.join(IMAGE_DIR, folder_name)
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(".png"):
                items.append((os.path.join(folder, f), label))
    return items


ALL_IMAGES = load_image_list()


# ---- 3. Model prediction, with the same preprocessing as training ----
def predict_real_prob(path):
    img = Image.open(path).convert("RGB").resize((128, 128))
    x = np.array(img, dtype="float32")[None, ...]   # [0, 255], NO division by 255
    return float(model.predict(x, verbose=0)[0, 0])  # sigmoid output = P(Real)


# ---- 4. Game state and display helpers ----
def new_state():
    return {
        "order": random.sample(ALL_IMAGES, N_ROUNDS),  # 10 distinct images
        "round": 0,                                    # index of the current image
        "user": 0,                                     # user's correct answers
        "model": 0,                                    # model's correct answers
    }


def scoreboard_html(s=None, answered=False):
    """Round counter, both scores and a progress bar."""
    if s is None:
        return ('<div class="scorecard"><div class="round">Ready when you are</div>'
                '<div class="hint">Press <b>New game</b> to start.</div></div>')
    done = s["round"] + (1 if answered else 0)          # rounds finished so far
    pct = 100 * done / N_ROUNDS
    return f"""
    <div class="scorecard">
      <div class="round">Round {s['round'] + 1} of {N_ROUNDS}</div>
      <div class="scores">
        <div class="score"><span class="who">You</span><span class="val">{s['user']}</span></div>
        <div class="score"><span class="who">Model</span><span class="val">{s['model']}</span></div>
      </div>
      <div class="bar"><div class="fill" style="width:{pct:.0f}%"></div></div>
    </div>"""


def summary_text(s):
    if s["user"] > s["model"]:
        verdict = "You beat the model! 🎉"
    elif s["user"] < s["model"]:
        verdict = "The model wins this time. 🤖"
    else:
        verdict = "It's a tie. 🤝"
    return (f"### Game over\n"
            f"**You:** {s['user']} / {N_ROUNDS} ({s['user'] / N_ROUNDS:.0%})  |  "
            f"**Model:** {s['model']} / {N_ROUNDS} ({s['model'] / N_ROUNDS:.0%})\n\n"
            f"{verdict}  Press **New game** to play again.")


PROMPT = "#### Is this face **Real** or **AI-generated**?\nLook closely, then make your guess."


# ---- 5. Event handlers ----
def start_game():
    s = new_state()
    path, _ = s["order"][0]
    return (s, path, scoreboard_html(s), PROMPT,
            gr.Button(interactive=True), gr.Button(interactive=True), gr.Button(interactive=False))


def make_guess(s, user_guess):
    path, label = s["order"][s["round"]]
    p = predict_real_prob(path)
    model_guess = int(p >= 0.5)
    confidence = p if model_guess == 1 else 1 - p

    user_ok = user_guess == label
    model_ok = model_guess == label
    s["user"] += int(user_ok)
    s["model"] += int(model_ok)

    feedback = (f"#### The answer: {LABEL[label]}\n\n"
                f"{'✅' if user_ok else '❌'} **You** said {LABEL[user_guess]}\n\n"
                f"{'✅' if model_ok else '❌'} **Model** said {LABEL[model_guess]} "
                f"({confidence:.0%} confident, P(Real) = {p:.2f})")

    last_round = s["round"] == N_ROUNDS - 1
    if last_round:
        feedback += "\n\n---\n" + summary_text(s)

    return (s, scoreboard_html(s, answered=True), feedback,
            gr.Button(interactive=False), gr.Button(interactive=False),
            gr.Button(interactive=not last_round))


def next_round(s):
    s["round"] += 1
    path, _ = s["order"][s["round"]]
    return (s, path, scoreboard_html(s), PROMPT,
            gr.Button(interactive=True), gr.Button(interactive=True), gr.Button(interactive=False))


# ---- 6. Styling (in Gradio 6, theme and css are passed to launch()) ----
THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")

CSS = """
.gradio-container { max-width: 980px !important; margin: 0 auto !important; }
#header { text-align: center; margin-bottom: 4px; }
#header h1 { margin-bottom: 2px; }
#author { text-align: center; opacity: 0.75; font-size: 0.9em; margin-bottom: 12px; }
.scorecard { border: 1px solid var(--border-color-primary, #e2e8f0); border-radius: 12px;
             padding: 14px 16px; background: var(--block-background-fill, #fff); }
.scorecard .round { font-weight: 600; font-size: 1.05em; margin-bottom: 8px; }
.scorecard .hint { opacity: 0.75; }
.scorecard .scores { display: flex; gap: 12px; margin-bottom: 10px; }
.scorecard .score { flex: 1; text-align: center; border-radius: 10px; padding: 8px 0;
                    background: var(--background-fill-secondary, #f1f5f9); }
.scorecard .who { display: block; font-size: 0.8em; opacity: 0.7; text-transform: uppercase;
                  letter-spacing: 0.05em; }
.scorecard .val { display: block; font-size: 1.6em; font-weight: 700; }
.scorecard .bar { height: 6px; border-radius: 3px; overflow: hidden;
                  background: var(--background-fill-secondary, #e2e8f0); }
.scorecard .fill { height: 100%; background: var(--color-accent, #6366f1); transition: width 0.3s; }
#feedback { min-height: 150px; }
#footer { text-align: center; opacity: 0.6; font-size: 0.85em; margin-top: 16px; }
"""


# ---- 7. Layout ----
with gr.Blocks(title="Real or AI? Face Challenge") as demo:
    gr.Markdown("# Real or AI? 🕵️\n"
                "Can you spot AI-generated faces better than a fine-tuned MobileNetV3Small?",
                elem_id="header")
    gr.Markdown(f"Built by {AUTHOR}", elem_id="author")

    with gr.Accordion("How to play", open=False):
        gr.Markdown(f"1. Press **New game**. You get {N_ROUNDS} faces, one per round.\n"
                    "2. Decide whether each face is **Real** or **AI-generated**.\n"
                    "3. After your guess, you see the true answer and the model's prediction.\n"
                    "4. Whoever gets more right after all rounds wins.\n\n"
                    "All faces come from the held-out test set, so the model has never seen them.")

    state = gr.State()

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            image = gr.Image(show_label=False, type="filepath", interactive=False, height=400)
        with gr.Column(scale=4):
            scoreboard = gr.HTML(scoreboard_html())
            feedback = gr.Markdown("", elem_id="feedback")
            with gr.Row():
                real_btn = gr.Button("👤 Real", variant="primary", interactive=False)
                ai_btn = gr.Button("🤖 AI-generated", variant="primary", interactive=False)
            with gr.Row():
                next_btn = gr.Button("Next round ➡", variant="secondary", interactive=False)
                new_btn = gr.Button("New game", variant="secondary")

    gr.Markdown("Model: MobileNetV3Small fine-tuned on "
                "real vs AI-generated faces, 76% test accuracy.", elem_id="footer")

    round_outputs = [state, image, scoreboard, feedback, real_btn, ai_btn, next_btn]
    guess_outputs = [state, scoreboard, feedback, real_btn, ai_btn, next_btn]

    new_btn.click(start_game, inputs=None, outputs=round_outputs)
    real_btn.click(lambda s: make_guess(s, 1), inputs=state, outputs=guess_outputs)
    ai_btn.click(lambda s: make_guess(s, 0), inputs=state, outputs=guess_outputs)
    next_btn.click(next_round, inputs=state, outputs=round_outputs)


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)