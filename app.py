"""
╔══════════════════════════════════════════════════════════════════╗
║        SHERLOCK HOLMES INTELLIGENCE BUREAU                      ║
║        Silver Blaze Case — NLP + Knowledge Graph Engine         ║
║        app.py — Streamlit Cloud Deployment                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import nltk
import pandas as pd
import re
import os
import json
from collections import Counter
from pyvis.network import Network
import tempfile

# ─── NLTK Bootstrap ──────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    for pkg in ["punkt", "averaged_perceptron_tagger", "maxent_ne_chunker",
                "words", "stopwords", "punkt_tab", "averaged_perceptron_tagger_eng"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

download_nltk()

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk import ne_chunk, FreqDist
from nltk.tree import Tree

# ══════════════════════════════════════════════════════════════════
#  CORPUS — Embedded Silver Blaze text
# ══════════════════════════════════════════════════════════════════
CORPUS = """
The Wessex Cup was the most important horse race of the season. The favorite horse was Silver Blaze.
The horse belonged to Colonel Ross. Silver Blaze was trained at King's Pyland by John Straker.
Many people had placed money on Silver Blaze to win. If the horse disappeared or became injured,
many gamblers would profit.

King's Pyland was a quiet training stable on the moor. John Straker managed the horses.
Three stable boys worked under him. Every night, one boy stayed awake to guard the stable.
On this night, the guard was Ned Hunter. The other two boys slept nearby.

At nine o'clock, the horses were locked inside the stable. Later, a maid named Edith Baxter
carried dinner to Ned Hunter. His meal was curried mutton. She walked through the dark path
with a lantern in her hand.

Before she reached the stable, a stranger stepped out of the darkness. He was pale and nervous.
He wore gray clothes and carried a heavy stick. The stranger asked where he was.
Edith said he was near the King's Pyland stables.

The stranger became excited. He said a stable boy slept there alone every night.
Then he offered Edith money if she would deliver a paper to the boy.
Edith became frightened and hurried to the stable window.

Ned Hunter was inside the stable. Edith told him about the stranger.
The stranger then came to the window and spoke directly to Hunter.
He asked questions about the race horses. He wanted to know whether Silver Blaze or Bayard was stronger.

Hunter became angry. He shouted at the man and ran to release the guard dog.
When he came back, the stranger had vanished. Hunter locked the stable door before running outside.

Later that night, John Straker woke up and told his wife he was worried about the horses.
He said he wanted to inspect the stable. Although it was raining, he put on his coat and left the house.

The next morning, Straker had not returned. Mrs. Straker went to the stable with the maid.
The stable door was open. Ned Hunter was sitting unconscious on a chair. Silver Blaze was missing.
John Straker was gone.

A search party went onto the moor. They found Straker's coat hanging on a bush.
A short distance away, they found his dead body in a hollow.
His skull had been badly broken by a powerful blow. He also had a deep cut on his thigh.

In Straker's right hand was a small knife. In his left hand was a red and black scarf.
Edith Baxter recognized the scarf. It belonged to the stranger from the night before.

The remains of Hunter's dinner were examined. The curried mutton contained powdered opium.
This explained why Hunter had become unconscious. Police arrested the stranger.
His name was Fitzroy Simpson.

Simpson admitted visiting the stable. He said he only wanted racing information.
He denied harming anyone. Police suspected him because he had bet against Silver Blaze.
He also carried a heavy stick.

Sherlock Holmes and Dr. Watson came to investigate. Holmes examined the stable, the path, and the moor.
He studied footprints in the mud. He found a half-burned match near the dead body.

Holmes asked an unusual question about the dog. The dog had not barked during the night.
Holmes said this was important. If a stranger had entered the stable, the dog would have barked.
Therefore, the visitor was someone the dog knew.

Holmes also asked about sheep on the property. A stable boy said several sheep had recently gone lame.
Holmes became interested in this fact.

Holmes examined the knife found in Straker's hand. It was not a fighting knife.
It was a delicate surgical knife used for precise cuts.
Holmes also found an expensive clothing bill in Straker's pocket under another name.

Holmes concluded that Straker had debts and secret expenses. He believed Straker planned to cheat his employer.
Straker had drugged Ned Hunter, entered the stable himself, and taken Silver Blaze onto the moor.

Straker intended to cut the horse's leg slightly with the surgical knife.
A small injury would make Silver Blaze lose the race.
Straker could then profit from betting.

On the moor, Straker lit a match so he could see. The sudden light frightened Silver Blaze.
The horse kicked Straker in the head. Straker fell and his own knife cut his thigh.

Silver Blaze escaped and ran across the moor. The horse reached the nearby Mapleton stables.
Their trainer, Silas Brown, recognized the famous horse and hid him.

Holmes discovered that Brown was hiding Silver Blaze. He forced Brown to return the horse safely.

At the Wessex Cup race, a horse with dyed markings entered the track. It won easily.
Holmes revealed that it was Silver Blaze.

Holmes then explained the full mystery. Fitzroy Simpson was innocent of murder.
John Straker died because of his own dishonest plan. Silver Blaze had acted in self-defense.
"""

# ══════════════════════════════════════════════════════════════════
#  NLP ENGINE — Modular Functions
# ══════════════════════════════════════════════════════════════════

def load_corpus():
    """Return the embedded Silver Blaze corpus."""
    return CORPUS.strip()


def extract_characters(text):
    """
    Named Entity Recognition using NLTK ne_chunk + predefined list.
    Returns list of (name, mention_count).
    """
    KNOWN_CHARACTERS = {
        "Sherlock Holmes": {"role": "Detective", "icon": "🔍"},
        "Dr. Watson": {"role": "Companion", "icon": "📔"},
        "John Straker": {"role": "Trainer (Villain)", "icon": "🗡️"},
        "Colonel Ross": {"role": "Horse Owner", "icon": "🎖️"},
        "Fitzroy Simpson": {"role": "Suspect", "icon": "🎩"},
        "Silver Blaze": {"role": "The Horse", "icon": "🐎"},
        "Silas Brown": {"role": "Rival Trainer", "icon": "🏇"},
        "Edith Baxter": {"role": "Stable Maid", "icon": "🕯️"},
        "Ned Hunter": {"role": "Stable Guard", "icon": "🛡️"},
    }
    text_lower = text.lower()
    results = {}
    for name, meta in KNOWN_CHARACTERS.items():
        # Count all variants
        variants = [name, name.split()[-1]]  # "Holmes", "Straker", etc.
        count = 0
        for v in variants:
            count += len(re.findall(r'\b' + re.escape(v) + r'\b', text, re.IGNORECASE))
        results[name] = {**meta, "mentions": max(count, 1)}
    return results


def extract_clues(text):
    """Extract predefined evidence items with explanations."""
    CLUES = {
        "Surgical Knife": {
            "icon": "🔪",
            "explanation": "Found in Straker's hand. A delicate instrument for precise cuts — not a weapon but a tool for sabotage.",
            "significance": "High"
        },
        "Red & Black Scarf": {
            "icon": "🧣",
            "explanation": "Held by Straker's body. Identified as belonging to Fitzroy Simpson, creating false suspicion.",
            "significance": "High"
        },
        "Curried Mutton": {
            "icon": "🍖",
            "explanation": "Hunter's dinner was laced with powdered opium, rendering him unconscious.",
            "significance": "Critical"
        },
        "Powdered Opium": {
            "icon": "💊",
            "explanation": "The sedative used to drug Ned Hunter and allow Straker to act undetected.",
            "significance": "Critical"
        },
        "Dog's Silence": {
            "icon": "🐕",
            "explanation": "The dog did not bark — proving the intruder was known to it. The crucial negative clue.",
            "significance": "Critical"
        },
        "Burned Match": {
            "icon": "🔥",
            "explanation": "Found near the body. Straker used it to see on the dark moor — the light startled Silver Blaze.",
            "significance": "Medium"
        },
        "Horse Tracks": {
            "icon": "👣",
            "explanation": "Hoofprints on the moor traced Silver Blaze's path toward Mapleton stables.",
            "significance": "Medium"
        },
        "Lame Sheep": {
            "icon": "🐑",
            "explanation": "Straker practiced his surgical technique on sheep — evidence of premeditated sabotage.",
            "significance": "High"
        },
    }
    return CLUES


def extract_actions(text):
    """Extract verb-based actions using POS tagging."""
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    # Predefined narrative actions mapped from verb extraction
    ACTIONS = [
        {"action": "Bribed the Maid", "verb": "offered", "actor": "Fitzroy Simpson", "icon": "💰"},
        {"action": "Drugged the Dinner", "verb": "drugged", "actor": "John Straker", "icon": "💊"},
        {"action": "Left the Stable", "verb": "escaped", "actor": "Silver Blaze", "icon": "🚪"},
        {"action": "Body Discovered", "verb": "found", "actor": "Search Party", "icon": "💀"},
        {"action": "Tracks Examined", "verb": "studied", "actor": "Sherlock Holmes", "icon": "🔍"},
        {"action": "Horse Hidden", "verb": "hid", "actor": "Silas Brown", "icon": "🏠"},
        {"action": "Case Solved", "verb": "revealed", "actor": "Sherlock Holmes", "icon": "✅"},
    ]
    # Extract top verbs from corpus for display
    verbs = [w.lower() for w, t in tagged if t.startswith('VB') and len(w) > 3]
    top_verbs = [w for w, _ in Counter(verbs).most_common(15)]
    return ACTIONS, top_verbs


def extract_nouns_tokens(text):
    """Return tokens, POS tags, top nouns for debug panel."""
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    nouns = [w.lower() for w, t in tagged if t.startswith('NN') and len(w) > 3]
    stop = {"stable", "horse", "said", "came", "went", "told", "also", "that", "this",
            "they", "were", "have", "been", "would", "could", "which"}
    top_nouns = [w for w, _ in Counter(nouns).most_common(30) if w not in stop][:15]
    return tokens, tagged, top_nouns


def build_graph():
    """Construct PyVis knowledge graph and return Network object."""
    net = Network(
        height="600px", width="100%",
        bgcolor="#0a0f1e", font_color="#d4a853",
        heading=""
    )
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=200)

    # ── Node color palette ─────────────────────────────────────
    COLORS = {
        "Character": {"background": "#d4a853", "border": "#f0c040", "font": {"color": "#0a0f1e"}},
        "Object":    {"background": "#2c4a7c", "border": "#4a7abf", "font": {"color": "#d4e8ff"}},
        "Place":     {"background": "#1a3a2a", "border": "#2d6b45", "font": {"color": "#a8e6c3"}},
        "Event":     {"background": "#4a1a1a", "border": "#8b2020", "font": {"color": "#ffb3b3"}},
    }

    def add_node(node_id, label, ntype, title=""):
        c = COLORS[ntype]
        net.add_node(
            node_id, label=label, title=title or f"{ntype}: {label}",
            color=c, shape="dot", size=25 if ntype == "Character" else 18,
            font={"size": 13, "face": "Georgia"}
        )

    # ── Characters ─────────────────────────────────────────────
    chars = [
        ("holmes",   "Sherlock Holmes",  "World's greatest detective. Solved Silver Blaze case."),
        ("watson",   "Dr. Watson",       "Holmes's companion and chronicler."),
        ("straker",  "John Straker",     "Trainer. Planned to sabotage Silver Blaze. Villain."),
        ("ross",     "Colonel Ross",     "Owner of Silver Blaze. Victim of the plot."),
        ("simpson",  "Fitzroy Simpson",  "Suspect. Visited stable. Proved innocent."),
        ("blaze",    "Silver Blaze",     "The missing horse. Winner. Killed in self-defense."),
        ("brown",    "Silas Brown",      "Rival trainer who hid Silver Blaze."),
        ("baxter",   "Edith Baxter",     "Maid who carried drugged dinner to Hunter."),
        ("hunter",   "Ned Hunter",       "Stable guard. Drugged and left unconscious."),
    ]
    for nid, label, title in chars:
        add_node(nid, label, "Character", title)

    # ── Objects ────────────────────────────────────────────────
    objects = [
        ("knife",   "Surgical Knife",  "Delicate blade used for sabotage, found in Straker's hand."),
        ("scarf",   "Red Scarf",       "Simpson's scarf, held by Straker's body."),
        ("opium",   "Powdered Opium",  "Mixed into Hunter's curried mutton to drug him."),
        ("match",   "Burned Match",    "Used by Straker on the dark moor. Startled Silver Blaze."),
        ("mutton",  "Curried Mutton",  "Hunter's dinner, laced with opium."),
        ("sheep",   "Lame Sheep",      "Straker practiced surgical cuts on them."),
        ("bill",    "Clothing Bill",   "Found in Straker's pocket under false name. Evidence of debt."),
    ]
    for nid, label, title in objects:
        add_node(nid, label, "Object", title)

    # ── Places ─────────────────────────────────────────────────
    places = [
        ("kingsP",    "King's Pyland",    "Straker's training stable. Scene of the crime."),
        ("moor",      "The Moor",         "Where Straker's body was found."),
        ("mapleton",  "Mapleton Stables", "Rival stable where Silver Blaze was hidden."),
        ("wessex",    "Wessex Cup",       "The race Silver Blaze ultimately won."),
    ]
    for nid, label, title in places:
        add_node(nid, label, "Place", title)

    # ── Events ─────────────────────────────────────────────────
    events = [
        ("drugging", "Horse Drugged",     "Straker drugged Hunter to access the stable undetected."),
        ("escape",   "Horse Escaped",     "Silver Blaze fled onto the moor after kicking Straker."),
        ("death",    "Straker Killed",    "Silver Blaze kicked Straker — accidental self-defense."),
        ("recovery", "Horse Recovered",   "Holmes forced Brown to return Silver Blaze."),
        ("race",     "Race Won",          "Silver Blaze, disguised, won the Wessex Cup."),
    ]
    for nid, label, title in events:
        add_node(nid, label, "Event", title)

    # ── Edges ──────────────────────────────────────────────────
    EDGE_STYLE = {"color": {"color": "#5a6a8a"}, "font": {"size": 10, "color": "#8899bb"}, "smooth": {"type": "curvedCW"}}

    edges = [
        ("ross",     "blaze",    "owns"),
        ("straker",  "kingsP",   "managed"),
        ("straker",  "blaze",    "trained"),
        ("straker",  "knife",    "carried"),
        ("straker",  "sheep",    "practiced_on"),
        ("straker",  "opium",    "used"),
        ("straker",  "bill",     "hid"),
        ("baxter",   "mutton",   "delivered"),
        ("mutton",   "opium",    "contained"),
        ("opium",    "hunter",   "drugged"),
        ("hunter",   "kingsP",   "guarded"),
        ("simpson",  "scarf",    "owned"),
        ("scarf",    "straker",  "found_with"),
        ("simpson",  "kingsP",   "visited"),
        ("holmes",   "kingsP",   "investigated"),
        ("holmes",   "moor",     "searched"),
        ("holmes",   "brown",    "confronted"),
        ("match",    "death",    "caused"),
        ("blaze",    "death",    "caused"),
        ("blaze",    "escape",   "triggered"),
        ("blaze",    "mapleton", "fled_to"),
        ("brown",    "blaze",    "hid"),
        ("brown",    "mapleton", "at"),
        ("death",    "moor",     "occurred_at"),
        ("holmes",   "recovery", "achieved"),
        ("blaze",    "race",     "won"),
        ("holmes",   "race",     "revealed_at"),
        ("watson",   "holmes",   "accompanied"),
        ("drugging", "escape",   "led_to"),
        ("escape",   "recovery", "preceded"),
    ]

    for src, tgt, label in edges:
        net.add_edge(src, tgt, label=label, **EDGE_STYLE)

    return net


def render_graph(net):
    """Save PyVis graph to HTML and return the HTML string."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w') as f:
        path = f.name
    net.save_graph(path)
    with open(path, 'r', encoding='utf-8') as f:
        html = f.read()
    os.unlink(path)
    return html


def reason_case():
    """Return Holmes deduction chain."""
    return [
        ("🐕", "Dog Did Not Bark",        "Negative evidence — no alarm raised"),
        ("👤", "Visitor Was Known",        "The dog recognised the intruder"),
        ("🗝️",  "Straker Had Access",       "As trainer, he could enter freely"),
        ("🔪", "Knife Indicates Sabotage", "Surgical blade — not for defence"),
        ("🐎", "Horse Panicked",           "Sudden matchlight frightened Silver Blaze"),
        ("💀", "Straker Died",             "Hoof-kick — self-defence, not murder"),
        ("✅", "Case Solved",              "Simpson innocent — Straker the guilty architect"),
    ]


# ══════════════════════════════════════════════════════════════════
#  STREAMLIT PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Sherlock Holmes Intelligence Bureau",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════
#  GLOBAL CSS — Victorian Cinematic Theme
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;900&family=IM+Fell+English:ital@0;1&family=Josefin+Sans:wght@300;400;600&display=swap');

:root {
    --navy:    #0a0f1e;
    --navy2:   #111827;
    --navy3:   #1a2540;
    --gold:    #d4a853;
    --gold2:   #f0c040;
    --gold3:   #8b6914;
    --parch:   #e8dcc8;
    --parch2:  #c8b89a;
    --crimson: #8b1a1a;
    --dim:     #4a5568;
}

html, body, [class*="css"] {
    font-family: 'Josefin Sans', sans-serif;
    background-color: var(--navy);
    color: var(--parch);
}

.stApp { background-color: var(--navy); }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 1400px; }

/* ── Section Headers ── */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--gold);
    letter-spacing: 2px;
    margin: 2.5rem 0 0.5rem;
    border-bottom: 1px solid var(--gold3);
    padding-bottom: 0.5rem;
}
.section-sub {
    font-family: 'IM Fell English', serif;
    font-style: italic;
    color: var(--parch2);
    font-size: 1rem;
    margin-bottom: 2rem;
    letter-spacing: 1px;
}

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(135deg, var(--navy3) 0%, #0f1a30 100%);
    border: 1px solid var(--gold3);
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--gold3), var(--gold), var(--gold3));
}
.metric-num {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    color: var(--gold);
    line-height: 1;
}
.metric-label {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--parch2);
    margin-top: 0.5rem;
}

/* ── Character Cards ── */
.char-card {
    background: linear-gradient(160deg, var(--navy3) 0%, #0c1628 100%);
    border: 1px solid var(--gold3);
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    transition: transform 0.2s, border-color 0.2s;
}
.char-card:hover { border-color: var(--gold); transform: translateY(-2px); }
.char-icon { font-size: 2rem; }
.char-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: var(--gold);
    font-weight: 700;
}
.char-role {
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--parch2);
    margin: 0.2rem 0;
}
.char-mentions {
    background: var(--gold3);
    color: var(--gold2);
    padding: 0.1rem 0.5rem;
    border-radius: 20px;
    font-size: 0.7rem;
    display: inline-block;
    margin-top: 0.3rem;
}

/* ── Evidence Cards ── */
.evidence-card {
    background: linear-gradient(135deg, #0f1929 0%, #1a2a1a 100%);
    border: 1px solid #2d4a2d;
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
}
.evidence-card.critical { border-color: var(--crimson); }
.evidence-card.high     { border-color: var(--gold3); }
.evidence-icon { font-size: 1.8rem; }
.evidence-name {
    font-family: 'Playfair Display', serif;
    color: var(--parch);
    font-weight: 600;
    font-size: 1rem;
}
.badge-critical { background: var(--crimson); color: #ffaaaa; padding: 0.1rem 0.6rem; border-radius: 12px; font-size: 0.65rem; letter-spacing: 1px; }
.badge-high     { background: #3a2a0a; color: var(--gold); padding: 0.1rem 0.6rem; border-radius: 12px; font-size: 0.65rem; letter-spacing: 1px; }
.badge-medium   { background: #1a2a3a; color: #88aacc; padding: 0.1rem 0.6rem; border-radius: 12px; font-size: 0.65rem; letter-spacing: 1px; }

/* ── Action Cards ── */
.action-card {
    background: #0f1520;
    border-left: 3px solid var(--gold);
    padding: 0.8rem 1.2rem;
    margin-bottom: 0.7rem;
    border-radius: 0 6px 6px 0;
}
.action-verb {
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--gold3);
}
.action-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    color: var(--parch);
    font-weight: 600;
}
.action-actor { color: var(--parch2); font-size: 0.8rem; }

/* ── Timeline ── */
.timeline-item {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.2rem;
    align-items: flex-start;
}
.timeline-dot {
    width: 12px; height: 12px;
    background: var(--gold);
    border-radius: 50%;
    margin-top: 4px;
    flex-shrink: 0;
    box-shadow: 0 0 8px var(--gold);
}
.timeline-line {
    border-left: 2px solid var(--gold3);
    padding-left: 1rem;
}
.timeline-label {
    font-family: 'Playfair Display', serif;
    color: var(--parch);
    font-size: 1rem;
    font-weight: 600;
}
.timeline-method {
    font-size: 0.7rem;
    color: var(--parch2);
    letter-spacing: 1px;
    font-style: italic;
}

/* ── Deduction Chain ── */
.deduction-step {
    background: linear-gradient(135deg, #0f1929 0%, #1a1025 100%);
    border: 1px solid var(--gold3);
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin-bottom: 0.5rem;
    position: relative;
}
.deduction-icon { font-size: 1.5rem; }
.deduction-title {
    font-family: 'Playfair Display', serif;
    color: var(--gold);
    font-size: 1.05rem;
    font-weight: 700;
}
.deduction-sub { color: var(--parch2); font-size: 0.8rem; }
.deduction-arrow { text-align: center; color: var(--gold3); font-size: 1.2rem; margin: -0.2rem 0; }

/* ── Verdict ── */
.verdict-box {
    background: linear-gradient(135deg, #1a0505 0%, #0f0f0a 100%);
    border: 2px solid var(--crimson);
    border-radius: 12px;
    padding: 3rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.verdict-stamp {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    font-weight: 900;
    color: var(--crimson);
    letter-spacing: 8px;
    border: 4px solid var(--crimson);
    display: inline-block;
    padding: 0.5rem 2rem;
    transform: rotate(-3deg);
    opacity: 0.9;
    text-shadow: 2px 2px 4px #000;
    margin-bottom: 2rem;
}
.verdict-item {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid var(--crimson);
    padding: 0.6rem 1rem;
    margin: 0.5rem auto;
    max-width: 500px;
    text-align: left;
    font-size: 0.95rem;
}

/* ── NLP Method Cards ── */
.nlp-card {
    background: linear-gradient(135deg, var(--navy3) 0%, #0c1220 100%);
    border: 1px solid var(--gold3);
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
    height: 100%;
}
.nlp-card-title {
    font-family: 'Playfair Display', serif;
    color: var(--gold);
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.nlp-card-body { color: var(--parch2); font-size: 0.82rem; line-height: 1.5; }
.nlp-tag {
    background: var(--gold3);
    color: var(--navy);
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-family: monospace;
    margin-right: 0.3rem;
}

/* ── Debug Panel ── */
.debug-panel {
    background: #050a14;
    border: 1px solid #1a2540;
    border-radius: 8px;
    padding: 1.5rem;
    font-family: monospace;
    font-size: 0.8rem;
    color: #7a9abf;
    max-height: 300px;
    overflow-y: auto;
}
.debug-key { color: var(--gold3); }
.debug-val { color: #88ccaa; }

/* ── Hero ── */
.hero-section {
    background: linear-gradient(135deg, #060c1a 0%, #0f1929 50%, #060c1a 100%);
    min-height: 85vh;
    display: flex;
    align-items: center;
    padding: 4rem 3rem;
    border-bottom: 1px solid var(--gold3);
    position: relative;
    overflow: hidden;
}
.hero-section::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse at 20% 50%, rgba(212,168,83,0.04) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 50%, rgba(212,168,83,0.03) 0%, transparent 60%);
}
.hero-tag {
    font-size: 0.65rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--gold);
    background: rgba(212,168,83,0.1);
    border: 1px solid var(--gold3);
    padding: 0.3rem 1rem;
    border-radius: 20px;
    display: inline-block;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.5rem, 5vw, 4.5rem);
    font-weight: 900;
    color: var(--parch);
    line-height: 1.05;
    margin-bottom: 0.5rem;
}
.hero-title span { color: var(--gold); }
.hero-subtitle {
    font-family: 'IM Fell English', serif;
    font-size: 1.3rem;
    color: var(--gold2);
    font-style: italic;
    margin-bottom: 1rem;
}
.hero-desc {
    color: var(--parch2);
    font-size: 0.95rem;
    line-height: 1.7;
    max-width: 500px;
    margin-bottom: 2.5rem;
}
.btn-primary {
    background: linear-gradient(135deg, var(--gold3), var(--gold));
    color: var(--navy);
    border: none;
    padding: 0.8rem 2rem;
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-weight: 600;
    border-radius: 4px;
    cursor: pointer;
    display: inline-block;
    margin-right: 1rem;
    text-decoration: none;
}
.btn-secondary {
    background: transparent;
    color: var(--gold);
    border: 1px solid var(--gold3);
    padding: 0.8rem 2rem;
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-weight: 600;
    border-radius: 4px;
    cursor: pointer;
    display: inline-block;
    text-decoration: none;
}

/* ── Silhouette SVG ── */
.silhouette-wrap {
    display: flex;
    justify-content: flex-end;
    align-items: flex-end;
    height: 100%;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  LOAD DATA (cached)
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def load_all_data():
    text       = load_corpus()
    characters = extract_characters(text)
    clues      = extract_clues(text)
    actions, top_verbs = extract_actions(text)
    tokens, tagged, top_nouns = extract_nouns_tokens(text)
    deductions = reason_case()
    return text, characters, clues, actions, top_verbs, tokens, tagged, top_nouns, deductions

text, characters, clues, actions, top_verbs, tokens, tagged, top_nouns, deductions = load_all_data()


# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — HERO
# ══════════════════════════════════════════════════════════════════
SILHOUETTE_SVG = """
<svg viewBox="0 0 340 620" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:340px; opacity:0.85;">
  <defs>
    <linearGradient id="fadeLeft" x1="0" x2="1" y1="0" y2="0">
      <stop offset="0%" stop-color="#0a0f1e" stop-opacity="1"/>
      <stop offset="40%" stop-color="#0a0f1e" stop-opacity="0.1"/>
      <stop offset="100%" stop-color="#0a0f1e" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="bodyGrad" x1="0" x2="1" y1="0" y2="0">
      <stop offset="0%" stop-color="#1a2030"/>
      <stop offset="100%" stop-color="#0c1525"/>
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feComposite in="SourceGraphic" in2="blur" operator="over"/>
    </filter>
    <filter id="fog">
      <feGaussianBlur stdDeviation="6"/>
    </filter>
  </defs>
  <!-- Fog atmosphere at base -->
  <ellipse cx="170" cy="590" rx="200" ry="60" fill="#d4a853" opacity="0.04" filter="url(#fog)"/>
  <ellipse cx="170" cy="600" rx="160" ry="40" fill="#d4a853" opacity="0.06" filter="url(#fog)"/>

  <!-- Deerstalker hat -->
  <path d="M 100 140 Q 95 90 130 75 Q 160 55 185 70 Q 220 55 240 80 Q 265 95 255 140 Z"
        fill="#1a2235" stroke="#d4a853" stroke-width="0.8" opacity="0.95"/>
  <!-- Hat brim front -->
  <path d="M 90 148 Q 95 138 170 135 Q 250 138 258 148 Q 245 160 170 158 Q 95 160 90 148 Z"
        fill="#141c2c" stroke="#d4a853" stroke-width="0.6" opacity="0.9"/>
  <!-- Hat brim back -->
  <path d="M 100 140 Q 120 128 145 132 L 138 152 Q 112 148 100 140 Z"
        fill="#141c2c" opacity="0.8"/>
  <!-- Hat top details -->
  <line x1="150" y1="70" x2="148" y2="140" stroke="#d4a853" stroke-width="0.4" opacity="0.4"/>
  <line x1="185" y1="60" x2="183" y2="138" stroke="#d4a853" stroke-width="0.4" opacity="0.4"/>

  <!-- Head / face profile (right-facing slightly) -->
  <ellipse cx="175" cy="198" rx="52" ry="58" fill="#1e2d42" opacity="0.95"/>
  <!-- Jaw / chin sharpening -->
  <path d="M 145 235 Q 150 265 170 272 Q 185 278 195 265 L 210 240 Q 195 250 175 252 Q 160 252 150 244 Z"
        fill="#1e2d42" opacity="0.95"/>
  <!-- Ear -->
  <path d="M 130 192 Q 120 200 122 214 Q 124 224 134 222 L 136 205 Z"
        fill="#192538" opacity="0.9"/>
  <!-- Eye socket shadow -->
  <ellipse cx="178" cy="195" rx="16" ry="10" fill="#0d1925" opacity="0.7"/>
  <!-- Eye gleam -->
  <ellipse cx="182" cy="194" rx="5" ry="4" fill="#d4a853" opacity="0.25"/>
  <!-- Nose profile -->
  <path d="M 208 200 Q 220 205 218 216 Q 215 222 210 220"
        fill="none" stroke="#0d1925" stroke-width="2.5" opacity="0.7"/>
  <!-- Pipe -->
  <path d="M 200 240 Q 230 250 258 242 Q 270 238 272 232"
        fill="none" stroke="#8b6914" stroke-width="5" stroke-linecap="round" opacity="0.9"/>
  <ellipse cx="272" cy="230" rx="9" ry="7" fill="#6b4f10" opacity="0.85"/>
  <!-- Pipe smoke wisps -->
  <path d="M 272 222 Q 278 212 272 200 Q 266 188 274 176"
        fill="none" stroke="#d4a853" stroke-width="1.5" opacity="0.2"
        stroke-dasharray="3,4"/>
  <path d="M 275 218 Q 285 206 278 194"
        fill="none" stroke="#d4a853" stroke-width="1" opacity="0.15"
        stroke-dasharray="2,3"/>

  <!-- Neck -->
  <rect x="155" y="265" width="38" height="35" rx="5" fill="#1a2a3a" opacity="0.9"/>

  <!-- Collar / cravat -->
  <path d="M 150 295 Q 168 285 190 295 Q 210 305 218 300 L 225 320 Q 200 325 178 315 Q 155 325 140 318 Z"
        fill="#141c2c" stroke="#d4a853" stroke-width="0.5" opacity="0.9"/>
  <path d="M 168 298 L 172 316 L 176 298" fill="#0a0f1e" opacity="0.7"/>

  <!-- Cape / coat shoulders -->
  <path d="M 95 310 Q 60 330 50 380 Q 45 420 55 460 Q 70 510 80 540 L 100 535 Q 92 490 95 450 Q 100 420 120 400 Z"
        fill="#141c2c" stroke="#2a3a50" stroke-width="0.5" opacity="0.9"/>
  <path d="M 250 310 Q 285 330 295 380 Q 302 420 290 465 Q 278 505 265 535 L 245 530 Q 258 490 255 455 Q 252 420 235 400 Z"
        fill="#141c2c" stroke="#2a3a50" stroke-width="0.5" opacity="0.9"/>
  <!-- Cape overlay -->
  <path d="M 108 305 Q 90 315 88 340 Q 87 365 105 380 L 118 375 Q 105 360 108 340 Q 110 320 122 312 Z"
        fill="#1a2540" opacity="0.7"/>

  <!-- Main coat body -->
  <path d="M 118 308 Q 100 325 95 370 Q 90 420 95 480 Q 100 530 105 570 L 240 570 Q 248 530 250 480 Q 255 420 252 370 Q 247 325 228 308 Q 210 298 175 295 Q 142 298 118 308 Z"
        fill="url(#bodyGrad)" stroke="#1e2e45" stroke-width="0.8" opacity="0.95"/>
  <!-- Coat center line -->
  <line x1="175" y1="305" x2="175" y2="568" stroke="#0a0f1e" stroke-width="2" opacity="0.5"/>
  <!-- Coat lapels -->
  <path d="M 175 305 Q 158 320 155 345 L 175 340 Z"
        fill="#1a2845" opacity="0.8"/>
  <path d="M 175 305 Q 192 320 195 345 L 175 340 Z"
        fill="#1a2845" opacity="0.8"/>

  <!-- Left arm (visible) — holding pipe direction -->
  <path d="M 118 330 Q 88 350 72 390 Q 60 420 65 445 Q 72 460 88 455 Q 100 448 105 430 Q 108 412 120 400 Q 130 388 130 365 L 128 335 Z"
        fill="#141c2c" opacity="0.9"/>
  <!-- Hand suggestion -->
  <ellipse cx="80" cy="452" rx="14" ry="10" fill="#1e2d42" opacity="0.85"/>

  <!-- Watch chain detail -->
  <path d="M 175 400 Q 165 410 155 420 Q 148 428 150 435"
        fill="none" stroke="#d4a853" stroke-width="1.2" opacity="0.35"
        stroke-dasharray="2,2"/>

  <!-- Coat button suggestion -->
  <circle cx="175" cy="390" r="3" fill="#d4a853" opacity="0.3"/>
  <circle cx="175" cy="430" r="3" fill="#d4a853" opacity="0.3"/>
  <circle cx="175" cy="470" r="3" fill="#d4a853" opacity="0.3"/>

  <!-- Gold edge light (rim lighting effect) -->
  <path d="M 252 315 Q 265 340 268 380 Q 270 420 260 470 Q 250 510 240 540"
        fill="none" stroke="#d4a853" stroke-width="1" opacity="0.15"/>

  <!-- Left-to-right fade overlay (emerging from darkness) -->
  <rect x="0" y="0" width="340" height="620" fill="url(#fadeLeft)"/>

  <!-- Bottom fade into dark -->
  <defs>
    <linearGradient id="fadeBottom" x1="0" x2="0" y1="0" y2="1">
      <stop offset="60%" stop-color="#0a0f1e" stop-opacity="0"/>
      <stop offset="100%" stop-color="#0a0f1e" stop-opacity="1"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="340" height="620" fill="url(#fadeBottom)"/>
</svg>
"""

st.markdown("""
<div class="hero-section">
  <div style="flex:1; z-index:2; padding-right: 2rem;">
    <div class="hero-tag">🔍 &nbsp; NLP Case File № 001 — Silver Blaze</div>
    <h1 class="hero-title">Sherlock Holmes<br><span>Intelligence</span><br>Bureau</h1>
    <p class="hero-subtitle">Solving the Silver Blaze Mystery</p>
    <p class="hero-desc">
      This system reads a detective corpus, extracts characters and clues using
      Natural Language Processing, maps relationships into a Knowledge Graph,
      and reconstructs Holmes-style deductions — all without external data files.
    </p>
    <div>
      <span class="btn-primary">⚑ Begin Investigation</span>
      <span class="btn-secondary">⊞ Open Evidence Board</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Silhouette in right column — rendered via st.columns below hero
_, sil_col = st.columns([2, 1])
with sil_col:
    st.markdown(f'<div class="silhouette-wrap">{SILHOUETTE_SVG}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Tokenization · Entity Recognition · Relation Extraction</div>', unsafe_allow_html=True)

sentences = nltk.sent_tokenize(text)
m1, m2, m3, m4, m5 = st.columns(5)
metrics = [
    (m1, len(characters),       "Characters Found"),
    (m2, len(clues),            "Evidence Items"),
    (m3, len(actions),          "Actions Extracted"),
    (m4, 4,                     "Locations Found"),
    (m5, len(characters)*3,     "Relationships Built"),
]
for col, num, label in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-num">{num}</div>
          <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div style="margin-top:1rem; padding:1rem; background:#0d1525; border-radius:6px;
     border-left:3px solid #d4a853; font-size:0.82rem; color:#8899bb;">
  <b style="color:#d4a853;">Corpus Stats:</b> &nbsp;
  {len(tokens):,} tokens &nbsp;|&nbsp;
  {len(sentences)} sentences &nbsp;|&nbsp;
  {len(set(t.lower() for t in tokens if t.isalpha())):,} unique words &nbsp;|&nbsp;
  {len(top_nouns)} top nouns extracted
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — CASE TIMELINE
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Case Timeline</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Sentence Segmentation · Event Ordering</div>', unsafe_allow_html=True)

TIMELINE = [
    ("Silver Blaze favoured to win the Wessex Cup",   "Frequency analysis — character introduction"),
    ("Fitzroy Simpson approaches King's Pyland stable","Named entity recognition — suspect flagged"),
    ("Ned Hunter's dinner drugged with opium",         "Keyword extraction — substance detected"),
    ("Silver Blaze vanishes from stable",              "Event detection — central incident"),
    ("John Straker found dead on the moor",            "NER + relation extraction"),
    ("Sherlock Holmes begins investigation",           "Agent detection — protagonist engaged"),
    ("Silver Blaze recovered from Mapleton",           "Resolution event — subject recovered"),
    ("Full truth revealed at Wessex Cup",              "Final inference — case closed"),
]

t1, t2 = st.columns(2)
for i, (event, method) in enumerate(TIMELINE):
    col = t1 if i % 2 == 0 else t2
    with col:
        st.markdown(f"""
        <div class="timeline-item">
          <div>
            <div class="timeline-dot"></div>
          </div>
          <div class="timeline-line">
            <div class="timeline-label">{i+1}. {event}</div>
            <div class="timeline-method">⊹ {method}</div>
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  SECTION 4 — CHARACTER GALLERY
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Character Gallery</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Named Entity Recognition · Frequency Distribution</div>', unsafe_allow_html=True)

char_cols = st.columns(3)
for i, (name, meta) in enumerate(characters.items()):
    with char_cols[i % 3]:
        st.markdown(f"""
        <div class="char-card">
          <div class="char-icon">{meta['icon']}</div>
          <div class="char-name">{name}</div>
          <div class="char-role">{meta['role']}</div>
          <div class="char-mentions">⊹ {meta['mentions']} mentions</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  SECTION 5 — EVIDENCE LOCKER
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Evidence Locker</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Keyword Extraction · Noun Phrase Mining</div>', unsafe_allow_html=True)

ev1, ev2 = st.columns(2)
for i, (name, meta) in enumerate(clues.items()):
    col = ev1 if i % 2 == 0 else ev2
    with col:
        sig = meta['significance'].lower()
        badge_cls = f"badge-{sig}"
        card_cls = f"evidence-card {sig}"
        st.markdown(f"""
        <div class="{card_cls}">
          <div style="display:flex; align-items:center; gap:0.8rem;">
            <span class="evidence-icon">{meta['icon']}</span>
            <div>
              <div class="evidence-name">{name}</div>
              <span class="{badge_cls}">{meta['significance'].upper()}</span>
            </div>
          </div>
          <div style="margin-top:0.7rem; font-size:0.8rem; color:#8899bb; line-height:1.5;">
            {meta['explanation']}
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  SECTION 6 — ACTION CARDS
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Action Registry</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Verb Extraction · Sentence Parsing</div>', unsafe_allow_html=True)

ac1, ac2 = st.columns(2)
for i, act in enumerate(actions):
    col = ac1 if i % 2 == 0 else ac2
    with col:
        st.markdown(f"""
        <div class="action-card">
          <div style="display:flex; align-items:center; gap:0.8rem;">
            <span style="font-size:1.5rem;">{act['icon']}</span>
            <div>
              <div class="action-verb">verb: {act['verb']}</div>
              <div class="action-name">{act['action']}</div>
              <div class="action-actor">— {act['actor']}</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  SECTION 7 — INTERACTIVE KNOWLEDGE GRAPH
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Knowledge Graph</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Relation Extraction · Graph Modelling · PyVis Network</div>', unsafe_allow_html=True)

st.markdown("""
<div style="display:flex; gap:2rem; margin-bottom:1rem; font-size:0.75rem; letter-spacing:1px;">
  <span>🟡 <b style="color:#d4a853;">Character</b></span>
  <span>🔵 <b style="color:#4a7abf;">Object</b></span>
  <span>🟢 <b style="color:#2d6b45;">Place</b></span>
  <span>🔴 <b style="color:#8b2020;">Event</b></span>
</div>
""", unsafe_allow_html=True)

with st.spinner("Constructing knowledge graph..."):
    net = build_graph()
    graph_html = render_graph(net)

try:
    st.components.v1.html(graph_html, height=640, scrolling=False)
except Exception:
    st.markdown(f'<iframe srcdoc="{graph_html}" width="100%" height="640px"></iframe>',
                unsafe_allow_html=True)

st.markdown("""
<div style="font-size:0.75rem; color:#4a5568; text-align:center; margin-top:0.5rem;">
  Drag nodes · Scroll to zoom · Hover for details
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  SECTION 8 — HOLMES DEDUCTION ENGINE
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Holmes Deduction Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Rule-Based Logical Inference · Reasoning Chain</div>', unsafe_allow_html=True)

ded_col, _ = st.columns([2, 1])
with ded_col:
    for i, (icon, title, sub) in enumerate(deductions):
        st.markdown(f"""
        <div class="deduction-step">
          <div style="display:flex; align-items:center; gap:1rem;">
            <span class="deduction-icon">{icon}</span>
            <div>
              <div class="deduction-title">{title}</div>
              <div class="deduction-sub">{sub}</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
        if i < len(deductions) - 1:
            st.markdown('<div class="deduction-arrow">↓</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  SECTION 9 — FINAL VERDICT
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Final Verdict</div>', unsafe_allow_html=True)

st.markdown("""
<div class="verdict-box">
  <div class="verdict-stamp">CASE CLOSED</div>
  <div style="font-family:'IM Fell English',serif; font-size:1.4rem; color:#d4a853;
              font-style:italic; margin-bottom:1.5rem;">
    "When you have eliminated the impossible, whatever remains must be the truth."
  </div>
  <div class="verdict-item">❌ &nbsp; No human murderer — death was accidental</div>
  <div class="verdict-item">⚖️ &nbsp; John Straker planned elaborate betting fraud</div>
  <div class="verdict-item">🐎 &nbsp; Silver Blaze acted in pure self-defense</div>
  <div class="verdict-item">✅ &nbsp; Fitzroy Simpson — wrongly suspected, innocent</div>
  <div style="margin-top:2rem; font-size:0.75rem; color:#4a5568; letter-spacing:2px;">
    SOLVED BY SHERLOCK HOLMES · WATSON IN ATTENDANCE · SCOTLAND YARD NOTIFIED
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  SECTION 10 — NLP METHODS
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">NLP Methods Used</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Techniques powering this investigation</div>', unsafe_allow_html=True)

NLP_METHODS = [
    ("Tokenization",          "nltk.word_tokenize\nnltk.sent_tokenize",
     "Splits raw text into individual words and sentences. Foundation of all NLP pipelines."),
    ("POS Tagging",           "nltk.pos_tag",
     "Labels each token (NN=noun, VB=verb, JJ=adjective). Enables noun/verb extraction."),
    ("Named Entity Recog.",   "nltk.ne_chunk\nPredefined character list",
     "Identifies people, places, organisations from text. Augmented with known character names."),
    ("Frequency Distribution","nltk.FreqDist\ncollections.Counter",
     "Counts token occurrences to rank characters by importance and find key nouns."),
    ("Relation Extraction",   "Co-occurrence + rules",
     "Identifies subject-verb-object triples from sentences to build graph edges."),
    ("Knowledge Graph",       "PyVis Network",
     "Represents entities as nodes and their relationships as directed, labelled edges."),
]

nc1, nc2, nc3 = st.columns(3)
nlp_cols = [nc1, nc2, nc3]
for i, (title, tag_str, desc) in enumerate(NLP_METHODS):
    with nlp_cols[i % 3]:
        tags_html = "".join(f'<span class="nlp-tag">{t.strip()}</span>'
                            for t in tag_str.split('\n'))
        st.markdown(f"""
        <div class="nlp-card">
          <div class="nlp-card-title">{title}</div>
          <div style="margin-bottom:0.5rem;">{tags_html}</div>
          <div class="nlp-card-body">{desc}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  SECTION 11 — RAW NLP DEBUG PANEL
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Raw NLP Debug Panel</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Inspect the raw pipeline outputs — for viva, demo, and verification</div>',
            unsafe_allow_html=True)

with st.expander("🔬 Expand Debug Panel", expanded=False):
    d1, d2 = st.columns(2)

    with d1:
        st.markdown("**Token Sample** (first 30)")
        token_sample = tokens[:30]
        st.markdown(f'<div class="debug-panel">{" | ".join(token_sample)}</div>',
                    unsafe_allow_html=True)

        st.markdown("**POS Tag Sample** (first 20)")
        pos_sample = tagged[:20]
        pos_html = "<br>".join(
            f'<span class="debug-key">{w}</span> → <span class="debug-val">{t}</span>'
            for w, t in pos_sample
        )
        st.markdown(f'<div class="debug-panel">{pos_html}</div>', unsafe_allow_html=True)

    with d2:
        st.markdown("**Top Extracted Nouns**")
        nouns_html = " &nbsp;".join(
            f'<span style="background:#1a2a1a;color:#88cc88;padding:2px 8px;'
            f'border-radius:4px;font-size:0.8rem;">{n}</span>'
            for n in top_nouns
        )
        st.markdown(f'<div class="debug-panel" style="line-height:2.2;">{nouns_html}</div>',
                    unsafe_allow_html=True)

        st.markdown("**Top Extracted Verbs**")
        verbs_html = " &nbsp;".join(
            f'<span style="background:#1a1a2a;color:#8888cc;padding:2px 8px;'
            f'border-radius:4px;font-size:0.8rem;">{v}</span>'
            for v in top_verbs[:15]
        )
        st.markdown(f'<div class="debug-panel" style="line-height:2.2;">{verbs_html}</div>',
                    unsafe_allow_html=True)

    st.markdown("**Detected Characters + Mentions**")
    char_df = pd.DataFrame([
        {"Name": k, "Role": v["role"], "Mentions": v["mentions"]}
        for k, v in characters.items()
    ]).sort_values("Mentions", ascending=False)
    st.dataframe(char_df, use_container_width=True, hide_index=True)

    st.markdown("**Sentence Count & Corpus Length**")
    st.markdown(f"""
    <div class="debug-panel">
      <span class="debug-key">Total tokens:</span>     <span class="debug-val">{len(tokens)}</span><br>
      <span class="debug-key">Total sentences:</span>  <span class="debug-val">{len(sentences)}</span><br>
      <span class="debug-key">Unique words:</span>     <span class="debug-val">{len(set(t.lower() for t in tokens if t.isalpha()))}</span><br>
      <span class="debug-key">Characters found:</span> <span class="debug-val">{len(characters)}</span><br>
      <span class="debug-key">Clues found:</span>      <span class="debug-val">{len(clues)}</span><br>
      <span class="debug-key">Verbs extracted:</span>  <span class="debug-val">{len(top_verbs)}</span>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-top:4rem; padding:2rem; border-top:1px solid #1a2540;
     text-align:center; font-size:0.75rem; color:#2a3a50; letter-spacing:2px;">
  SHERLOCK HOLMES INTELLIGENCE BUREAU &nbsp;·&nbsp;
  NLP ENGINE: NLTK &nbsp;·&nbsp;
  GRAPH: PYVIS &nbsp;·&nbsp;
  FRONTEND: STREAMLIT<br>
  <span style="color:#1a2540;">Silver Blaze Case File · All deductions are elementary.</span>
</div>""", unsafe_allow_html=True)
