import hashlib
import random

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image

MODEL_PATH = "wighets/best.onnx"
CLASS_NAMES = ["Paper", "Rock", "Scissors"]  # order from best.pt
IMG_SIZE = 640
CONF_THRESHOLD = 0.25

MOVES = ["Rock", "Paper", "Scissors"]
BEATS = {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}
EMOJI = {"Rock": "✊", "Paper": "✋", "Scissors": "✌️"}
LABELS = {"win": "You Win! 🎉", "draw": "Draw 🤝", "lose": "Computer Wins! 🤖"}
HISTORY_ROWS = 10

st.set_page_config(page_title="Rock Paper Scissors", page_icon="✊", layout="wide")

st.markdown("""
<style>
  .stApp {
    background: repeating-linear-gradient(45deg, #ffd60a 0 30px, #ffc300 30px 60px);
  }
  .stApp, .stApp p, .stApp label { color: #111; }
  .title {
    display: inline-block;
    background: #111; color: #ffd60a;
    padding: 10px 28px; border-radius: 14px;
    box-shadow: 6px 6px 0 #fffdf2;
    transform: rotate(-2deg);
    font: 900 44px "Arial Black", Arial, sans-serif;
    margin: 0 0 16px;
  }
  .card {
    background: #fffdf2; border: 4px solid #111; border-radius: 18px;
    box-shadow: 6px 6px 0 #111; padding: 14px; text-align: center;
    font-family: "Arial Black", Arial, sans-serif; margin-bottom: 18px;
  }
  .label { font-size: 13px; letter-spacing: 1px; text-transform: uppercase; opacity: .6; }
  .value { font-size: 44px; line-height: 1.1; }
  .win { background: #2ecc71; }
  .draw { background: #ffd60a; }
  .lose { background: #ff4d4d; color: #fffdf2; }
  .hand { font-size: 72px; line-height: 1.2; }
  .vs {
    display: inline-block; background: #111; color: #ffd60a; border-radius: 50%;
    width: 52px; height: 52px; line-height: 52px; transform: rotate(-8deg);
  }
  .result { font-size: 26px; border: 4px solid #111; border-radius: 12px; padding: 12px; margin-top: 12px; }
  .chip {
    display: inline-block; padding: 6px 14px; margin: 4px;
    border: 3px solid #111; border-radius: 999px;
  }
  [data-testid="stCameraInput"] {
    background: #fffdf2; border: 4px solid #111; border-radius: 18px;
    box-shadow: 6px 6px 0 #111; padding: 10px;
  }
  .stButton button {
    background: #111; color: #ffd60a; border: 4px solid #111; border-radius: 14px;
    font-family: "Arial Black", Arial, sans-serif; box-shadow: 6px 6px 0 #fffdf2;
  }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


def detect_move(image):
    """Letterbox to 640x640 like YOLO, run the model, return (move, conf) of the best box."""
    img = image.convert("RGB")
    scale = min(IMG_SIZE / img.width, IMG_SIZE / img.height)
    w, h = round(img.width * scale), round(img.height * scale)
    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (114, 114, 114))
    canvas.paste(img.resize((w, h), Image.BILINEAR), ((IMG_SIZE - w) // 2, (IMG_SIZE - h) // 2))

    tensor = (np.asarray(canvas, dtype=np.float32) / 255).transpose(2, 0, 1)[None]
    session = load_model()
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]  # [4 + classes, boxes]

    scores = output[4:]
    cls, box = np.unravel_index(scores.argmax(), scores.shape)
    conf = float(scores[cls, box])
    return (CLASS_NAMES[cls], conf) if conf > CONF_THRESHOLD else (None, conf)


def new_game():
    st.session_state.score = {"win": 0, "draw": 0, "lose": 0}
    st.session_state.history = []
    st.session_state.last = None          # (player, computer, outcome)
    st.session_state.photo_id = None
    st.session_state.camera_key = st.session_state.get("camera_key", 0) + 1


def play_round(player):
    computer = random.choice(MOVES)
    if player == computer:
        outcome = "draw"
    elif BEATS[player] == computer:
        outcome = "win"
    else:
        outcome = "lose"
    st.session_state.score[outcome] += 1
    st.session_state.history.insert(0, (player, computer, outcome))
    del st.session_state.history[HISTORY_ROWS:]
    st.session_state.last = (player, computer, outcome)


if "score" not in st.session_state:
    new_game()

st.markdown('<div style="text-align:center"><div class="title">✊ ROCK PAPER SCISSORS ✌️</div></div>',
            unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")

with left:
    photo = st.camera_input("Show your hand and take a photo to play",
                            key=f"camera_{st.session_state.camera_key}")

    if photo is not None:
        photo_id = hashlib.md5(photo.getvalue()).hexdigest()
        if photo_id != st.session_state.photo_id:   # only score each photo once
            st.session_state.photo_id = photo_id
            move, conf = detect_move(Image.open(photo))
            if move:
                play_round(move)
            else:
                st.session_state.last = (None, None, "none")

with right:
    score = st.session_state.score
    cols = st.columns(3)
    for col, (label, key) in zip(cols, [("You", "win"), ("Draws", "draw"), ("Computer", "lose")]):
        col.markdown(f'<div class="card {key}"><div class="label">{label}</div>'
                     f'<div class="value">{score[key]}</div></div>', unsafe_allow_html=True)

    player, computer, outcome = st.session_state.last or (None, None, None)
    if outcome == "none":
        result, cls = "No hand detected 👀", ""
    elif outcome:
        result, cls = LABELS[outcome], outcome
    else:
        result, cls = "Take a photo to play", ""

    st.markdown(f"""
    <div class="card">
      <div style="display:flex;justify-content:space-around;align-items:center">
        <div><div class="label">You</div><div class="hand">{EMOJI.get(player, "❔")}</div>{player or "-"}</div>
        <div class="vs">VS</div>
        <div><div class="label">Computer</div><div class="hand">{EMOJI.get(computer, "🤖")}</div>{computer or "-"}</div>
      </div>
      <div class="result {cls}">{result}</div>
    </div>
    """, unsafe_allow_html=True)

    st.button("↺ RESET", on_click=new_game, use_container_width=True)

chips = "".join(f'<span class="chip {o}">{EMOJI[p]} vs {EMOJI[c]}</span>'
                for p, c, o in st.session_state.history)
st.markdown(f'<div class="card" style="text-align:left"><div class="label">History</div>'
            f'{chips or "No rounds played yet"}</div>', unsafe_allow_html=True)
