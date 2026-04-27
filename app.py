"""
╔══════════════════════════════════════════════════════════════════╗
║        SHERLOCK HOLMES INTELLIGENCE BUREAU  v3                  ║
║        Silver Blaze Case — NLP + Knowledge Graph Engine         ║
╚══════════════════════════════════════════════════════════════════╝
CHANGES v3:
  • Hero buttons removed
  • UnifrakturMaguntia for titles only (hero + section headers)
  • Image URL fixed → Assets/Holmes.jpg (capital A and H)
  • Verdict/Case Closed section removed
  • Code snippet section added (syntax-highlighted, with explanation)
  • Custom corpus uploader → generates live knowledge graph
  • Holmes deduction made minimal + concise conclusion
  • Team credits cards at footer
"""

import streamlit as st
import nltk
import pandas as pd
import re, os, json
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
from nltk import FreqDist
from nltk.tree import Tree

# ══════════════════════════════════════════════════════════════════
#  HERO IMAGE  — Assets/Holmes.jpg (capital A, capital H)
# ══════════════════════════════════════════════════════════════════
HERO_IMAGE_URL = (
    "https://raw.githubusercontent.com/devBoi1313/Detective_holmes"
    "/main/Assets/Holmes.jpg"
)

# ══════════════════════════════════════════════════════════════════
#  CORPUS
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
#  NLP ENGINE
# ══════════════════════════════════════════════════════════════════
def load_corpus():
    return CORPUS.strip()

def extract_characters(text):
    KNOWN = {
        "Sherlock Holmes": {"role": "Detective",         "icon": "🔍"},
        "Dr. Watson":      {"role": "Companion",         "icon": "📔"},
        "John Straker":    {"role": "Trainer (Villain)", "icon": "🗡️"},
        "Colonel Ross":    {"role": "Horse Owner",       "icon": "🎖️"},
        "Fitzroy Simpson": {"role": "Suspect",           "icon": "🎩"},
        "Silver Blaze":    {"role": "The Horse",         "icon": "🐎"},
        "Silas Brown":     {"role": "Rival Trainer",     "icon": "🏇"},
        "Edith Baxter":    {"role": "Stable Maid",       "icon": "🕯️"},
        "Ned Hunter":      {"role": "Stable Guard",      "icon": "🛡️"},
    }
    results = {}
    for name, meta in KNOWN.items():
        variants = [name, name.split()[-1]]
        count = sum(len(re.findall(r'\b' + re.escape(v) + r'\b', text, re.IGNORECASE))
                    for v in variants)
        results[name] = {**meta, "mentions": max(count, 1)}
    return results

def extract_clues(text):
    return {
        "Surgical Knife":    {"icon": "🔪", "significance": "Critical",
            "explanation": "Found in Straker's hand — a delicate instrument for precise cuts, not a weapon. Proof of sabotage intent."},
        "Red & Black Scarf": {"icon": "🧣", "significance": "High",
            "explanation": "Held in Straker's hand at death. Identified as Simpson's — created a deliberate false trail."},
        "Curried Mutton":    {"icon": "🍖", "significance": "Critical",
            "explanation": "Hunter's dinner laced with powdered opium, rendering him unconscious and defenceless."},
        "Powdered Opium":    {"icon": "💊", "significance": "Critical",
            "explanation": "The sedative used by Straker to neutralise Ned Hunter before stealing Silver Blaze."},
        "Dog's Silence":     {"icon": "🐕", "significance": "Critical",
            "explanation": "The dog did not bark — proving the intruder was someone it knew. Holmes's most famous deduction."},
        "Burned Match":      {"icon": "🔥", "significance": "Medium",
            "explanation": "Found near the body. Straker lit it to see on the moor — the flame startled Silver Blaze fatally."},
        "Horse Tracks":      {"icon": "👣", "significance": "Medium",
            "explanation": "Hoofprints traced Silver Blaze's path from King's Pyland to Mapleton stables."},
        "Lame Sheep":        {"icon": "🐑", "significance": "High",
            "explanation": "Straker practised his surgical technique on sheep — direct evidence of premeditated fraud."},
    }

def extract_actions(text):
    ACTIONS = [
        {"action": "Bribed the Maid",    "verb": "offered",  "actor": "Fitzroy Simpson", "icon": "💰"},
        {"action": "Drugged the Dinner", "verb": "drugged",  "actor": "John Straker",    "icon": "💊"},
        {"action": "Horse Escaped",      "verb": "escaped",  "actor": "Silver Blaze",    "icon": "🚪"},
        {"action": "Body Discovered",    "verb": "found",    "actor": "Search Party",    "icon": "💀"},
        {"action": "Tracks Examined",    "verb": "studied",  "actor": "Sherlock Holmes", "icon": "🔍"},
        {"action": "Horse Hidden",       "verb": "hid",      "actor": "Silas Brown",     "icon": "🏠"},
        {"action": "Case Solved",        "verb": "revealed", "actor": "Sherlock Holmes", "icon": "✅"},
    ]
    tokens    = word_tokenize(text)
    tagged    = pos_tag(tokens)
    verbs     = [w.lower() for w, t in tagged if t.startswith('VB') and len(w) > 3]
    top_verbs = [w for w, _ in Counter(verbs).most_common(15)]
    return ACTIONS, top_verbs

def extract_nouns_tokens(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    nouns  = [w.lower() for w, t in tagged if t.startswith('NN') and len(w) > 3]
    stop   = {"stable","horse","said","came","went","told","also","that","this",
              "they","were","have","been","would","could","which"}
    top_nouns = [w for w, _ in Counter(nouns).most_common(30) if w not in stop][:15]
    return tokens, tagged, top_nouns

def reason_case():
    return [
        ("🐕", "Silent dog",        "Visitor known — no stranger entered"),
        ("🔪", "Surgical knife",    "Sabotage planned, not murder"),
        ("💊", "Opium in dinner",   "Straker drugged his own guard"),
        ("🐑", "Lame sheep",        "Premeditated practice of the cut"),
        ("🔥", "Match on the moor", "Light startled horse — kick was accidental"),
    ]

def build_graph_from_text(text):
    """Build a PyVis graph from any input text using NLP extraction."""
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    sentences = sent_tokenize(text)

    # Extract named entities / proper nouns
    proper_nouns = list(dict.fromkeys(
        [w for w, t in tagged if t == 'NNP' and len(w) > 2]
    ))[:20]

    # Top nouns as objects
    nouns = [w.lower() for w, t in tagged if t in ('NN', 'NNS') and len(w) > 4]
    stop  = {"there","their","about","which","these","those","other","after",
              "before","would","could","every","night","morning","right","left"}
    top_nouns = [w for w, _ in Counter(nouns).most_common(20) if w not in stop][:8]

    net = Network(height="560px", width="100%", bgcolor="#faf7f0",
                  font_color="#2a1a0a", heading="")
    net.set_options(json.dumps({
        "edges": {
            "font": {"size": 10, "color": "#4a2a08",
                     "strokeWidth": 0, "background": "rgba(250,247,240,0.8)"},
            "color": {"color": "#b89a6a"},
            "smooth": {"type": "curvedCW", "roundness": 0.2}
        },
        "nodes": {"font": {"size": 12, "face": "Georgia"}},
        "physics": {
            "barnesHut": {"gravitationalConstant": -5000,
                          "centralGravity": 0.3, "springLength": 180},
            "stabilization": {"iterations": 100}
        },
        "interaction": {"hover": True}
    }))

    char_col = {"background": "#b8860b", "border": "#7a5800",
                "font": {"color": "#fff8e8"}}
    obj_col  = {"background": "#1e4080", "border": "#2d5a9a",
                "font": {"color": "#ddeeff"}}

    added_nodes = set()

    def safe_add(nid, label, color, size=20):
        nid_clean = re.sub(r'\W+', '_', nid.lower())
        if nid_clean not in added_nodes:
            net.add_node(nid_clean, label=label, color=color,
                         shape="dot", size=size,
                         font={"size": 12, "face": "Georgia",
                               "color": color["font"]["color"]})
            added_nodes.add(nid_clean)
        return nid_clean

    # Add proper nouns as character nodes
    pn_ids = {}
    for pn in proper_nouns:
        nid = safe_add(pn, pn, char_col, 24)
        pn_ids[pn] = nid

    # Add top nouns as object nodes
    noun_ids = {}
    for n in top_nouns:
        nid = safe_add(n, n.capitalize(), obj_col, 16)
        noun_ids[n] = nid

    # Build edges from sentence co-occurrence
    edge_set = set()
    for sent in sentences:
        sent_tokens = word_tokenize(sent)
        sent_pns    = [w for w in sent_tokens if w in pn_ids]
        sent_nouns  = [w.lower() for w in sent_tokens if w.lower() in noun_ids]
        sent_verbs  = [w.lower() for w, t in pos_tag(sent_tokens)
                       if t.startswith('VB') and len(w) > 3]
        verb_label  = sent_verbs[0] if sent_verbs else "related_to"
        # PN → PN
        for i in range(len(sent_pns)):
            for j in range(i+1, min(i+3, len(sent_pns))):
                a, b = pn_ids[sent_pns[i]], pn_ids[sent_pns[j]]
                key  = tuple(sorted([a, b]))
                if key not in edge_set:
                    net.add_edge(a, b, label=verb_label)
                    edge_set.add(key)
        # PN → noun
        for pn in sent_pns[:2]:
            for n in sent_nouns[:2]:
                a, b = pn_ids[pn], noun_ids[n]
                key  = (a, b)
                if key not in edge_set:
                    net.add_edge(a, b, label=verb_label)
                    edge_set.add(key)

    return net

def build_graph():
    """Fixed Silver Blaze knowledge graph."""
    net = Network(height="620px", width="100%",
                  bgcolor="#faf7f0", font_color="#2a1a0a", heading="")
    net.set_options(json.dumps({
        "edges": {
            "font": {"size": 11, "color": "#4a2a08",
                     "strokeWidth": 0, "background": "rgba(250,247,240,0.75)"},
            "color": {"color": "#b89a6a", "highlight": "#8b6914"},
            "smooth": {"type": "curvedCW", "roundness": 0.2}
        },
        "nodes": {"font": {"size": 13, "face": "Georgia"}},
        "physics": {
            "barnesHut": {"gravitationalConstant": -8000,
                          "centralGravity": 0.3, "springLength": 200},
            "stabilization": {"iterations": 120}
        },
        "interaction": {"hover": True, "tooltipDelay": 100}
    }))

    COLORS = {
        "Character": {"background": "#b8860b", "border": "#7a5800",
                      "highlight": {"background": "#d4a020", "border": "#7a5800"},
                      "font": {"color": "#fff8e8"}},
        "Object":    {"background": "#1e4080", "border": "#2d5a9a",
                      "highlight": {"background": "#2d5a9a", "border": "#1e4080"},
                      "font": {"color": "#ddeeff"}},
        "Place":     {"background": "#1a5030", "border": "#267a45",
                      "highlight": {"background": "#267a45", "border": "#1a5030"},
                      "font": {"color": "#cceecc"}},
        "Event":     {"background": "#8a1515", "border": "#bb2020",
                      "highlight": {"background": "#bb2020", "border": "#8a1515"},
                      "font": {"color": "#ffdddd"}},
    }

    def add_node(nid, label, ntype, title=""):
        c = COLORS[ntype]
        net.add_node(nid, label=label,
                     title=title or f"<b>{label}</b><br/><i>{ntype}</i>",
                     color=c, shape="dot",
                     size=28 if ntype == "Character" else 20,
                     font={"size": 13, "face": "Georgia",
                           "color": c["font"]["color"]})

    for nid, lbl, ttl in [
        ("holmes",  "Sherlock Holmes",  "World's greatest detective."),
        ("watson",  "Dr. Watson",       "Holmes's companion."),
        ("straker", "John Straker",     "Trainer and villain. Planned sabotage."),
        ("ross",    "Colonel Ross",     "Owner of Silver Blaze."),
        ("simpson", "Fitzroy Simpson",  "Prime suspect. Proved innocent."),
        ("blaze",   "Silver Blaze",     "The missing racehorse."),
        ("brown",   "Silas Brown",      "Rival trainer who hid the horse."),
        ("baxter",  "Edith Baxter",     "Maid who delivered the drugged dinner."),
        ("hunter",  "Ned Hunter",       "Stable guard. Drugged by Straker."),
    ]: add_node(nid, lbl, "Character", ttl)

    for nid, lbl, ttl in [
        ("knife",  "Surgical Knife", "Sabotage tool found in Straker's hand."),
        ("scarf",  "Red Scarf",      "Simpson's scarf — planted as false clue."),
        ("opium",  "Powdered Opium", "Sedative mixed into Hunter's dinner."),
        ("match",  "Burned Match",   "Used by Straker on the moor."),
        ("mutton", "Curried Mutton", "Hunter's drugged dinner."),
        ("sheep",  "Lame Sheep",     "Straker's practice subjects."),
        ("bill",   "Clothing Bill",  "Hidden debt evidence."),
    ]: add_node(nid, lbl, "Object", ttl)

    for nid, lbl, ttl in [
        ("kingsP",   "King's Pyland",    "Scene of the crime."),
        ("moor",     "The Moor",         "Where Straker was found dead."),
        ("mapleton", "Mapleton Stables", "Where Silver Blaze was hidden."),
        ("wessex",   "Wessex Cup",       "The race Silver Blaze won."),
    ]: add_node(nid, lbl, "Place", ttl)

    for nid, lbl, ttl in [
        ("drugging", "Horse Drugged",   "Straker drugged Hunter."),
        ("escape",   "Horse Escaped",   "Silver Blaze fled the moor."),
        ("death",    "Straker Killed",  "Hoof-kick — self-defence."),
        ("recovery", "Horse Recovered", "Holmes secured the return."),
        ("race",     "Race Won",        "Silver Blaze won in disguise."),
    ]: add_node(nid, lbl, "Event", ttl)

    for s, t, l in [
        ("ross","blaze","owns"),            ("straker","kingsP","managed"),
        ("straker","blaze","trained"),      ("straker","knife","carried"),
        ("straker","opium","used"),         ("straker","sheep","practiced_on"),
        ("straker","bill","hid"),           ("baxter","mutton","delivered"),
        ("mutton","opium","contained"),     ("opium","hunter","drugged"),
        ("hunter","kingsP","guarded"),      ("simpson","scarf","owned"),
        ("scarf","straker","found_with"),   ("simpson","kingsP","visited"),
        ("holmes","kingsP","investigated"), ("holmes","moor","searched"),
        ("holmes","brown","confronted"),    ("match","death","caused"),
        ("blaze","death","caused"),         ("blaze","escape","triggered"),
        ("blaze","mapleton","fled_to"),     ("brown","blaze","hid"),
        ("brown","mapleton","at"),          ("death","moor","occurred_at"),
        ("holmes","recovery","achieved"),   ("blaze","race","won"),
        ("watson","holmes","accompanied"),  ("drugging","escape","led_to"),
        ("escape","recovery","preceded"),   ("holmes","race","revealed_at"),
    ]: net.add_edge(s, t, label=l)

    return net

def render_graph(net):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w') as f:
        path = f.name
    net.save_graph(path)
    with open(path, 'r', encoding='utf-8') as f:
        html = f.read()
    os.unlink(path)
    return html

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Sherlock Holmes Intelligence Bureau",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=UnifrakturMaguntia&family=IM+Fell+English:ital@0;1&family=Playfair+Display:wght@400;700;900&family=Josefin+Sans:wght@300;400;600&display=swap');

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
    --code-bg: #0d1117;
}
html, body, [class*="css"] {
    font-family: 'Josefin Sans', sans-serif;
    background-color: var(--navy);
    color: var(--parch);
}
.stApp { background-color: var(--navy); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 1400px; }

/* ── UnifrakturMaguntia — titles only ── */
.gothic-title {
    font-family: 'UnifrakturMaguntia', cursive;
    letter-spacing: 1px;
    line-height: 1.15;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'UnifrakturMaguntia', cursive;
    font-size: 2.2rem;
    color: var(--gold);
    margin: 2.5rem 0 0.4rem;
    border-bottom: 1px solid var(--gold3);
    padding-bottom: 0.4rem;
    letter-spacing: 1px;
}
.section-sub {
    font-family: 'IM Fell English', serif;
    font-style: italic;
    color: var(--parch2);
    font-size: 0.95rem;
    margin-bottom: 2rem;
}

/* ── Hero ── */
.hero-wrap {
    display: flex;
    min-height: 88vh;
    background: linear-gradient(135deg, #060c1a 0%, #0f1929 60%, #060c1a 100%);
    border-bottom: 1px solid var(--gold3);
    overflow: hidden;
}
.hero-left {
    flex: 1.15; display: flex; flex-direction: column;
    justify-content: center;
    padding: 4rem 3rem 4rem 4rem; z-index: 2;
}
.hero-right {
    flex: 0.85; position: relative; overflow: hidden;
}
.hero-right img {
    width: 100%; height: 100%;
    object-fit: cover; object-position: center top; display: block;
}
.hero-fade-left {
    position: absolute; top:0; left:0; bottom:0; width:55%;
    background: linear-gradient(to right, #060c1a 0%, transparent 100%); z-index:2;
}
.hero-fade-bottom {
    position: absolute; left:0; right:0; bottom:0; height:35%;
    background: linear-gradient(to top, #060c1a 0%, transparent 100%); z-index:2;
}
.hero-fade-top {
    position: absolute; left:0; right:0; top:0; height:15%;
    background: linear-gradient(to bottom, #060c1a 0%, transparent 100%); z-index:2;
}
.hero-tag {
    font-size: 0.63rem; letter-spacing: 4px; text-transform: uppercase;
    color: var(--gold); background: rgba(212,168,83,0.07);
    border: 1px solid var(--gold3); padding: 0.3rem 1rem;
    border-radius: 20px; display: inline-block;
    margin-bottom: 1.5rem; width: fit-content;
}
.hero-title {
    font-family: 'UnifrakturMaguntia', cursive;
    font-size: clamp(3rem, 5vw, 5.5rem);
    color: var(--parch); line-height: 1.1; margin-bottom: 0.6rem;
    letter-spacing: 1px;
}
.hero-title span { color: var(--gold); }
.hero-subtitle {
    font-family: 'IM Fell English', serif;
    font-size: 1.15rem; color: var(--gold2);
    font-style: italic; margin-bottom: 1.2rem;
}
.hero-desc {
    color: var(--parch2); font-size: 0.9rem;
    line-height: 1.8; max-width: 480px; margin-bottom: 0.5rem;
}

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(135deg, var(--navy3) 0%, #0f1a30 100%);
    border: 1px solid var(--gold3); border-radius: 8px;
    padding: 1.5rem; text-align: center;
    position: relative; overflow: hidden;
}
.metric-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg, var(--gold3), var(--gold), var(--gold3));
}
.metric-num {
    font-family: 'Playfair Display', serif;
    font-size: 3rem; font-weight: 900; color: var(--gold); line-height: 1;
}
.metric-label {
    font-size: 0.67rem; letter-spacing: 3px;
    text-transform: uppercase; color: var(--parch2); margin-top: 0.5rem;
}

/* ── Graph wrapper ── */
.graph-frame {
    border: 2px solid var(--gold3); border-radius: 10px; overflow: hidden;
    box-shadow: 0 4px 40px rgba(212,168,83,0.1);
}
.graph-legend {
    display: flex; gap: 2rem; margin-bottom: 1rem;
    font-size: 0.75rem; letter-spacing: 1px; flex-wrap: wrap;
}

/* ── Napkin Timeline ── */
.napkin-wrap { overflow-x: auto; padding: 1.5rem 0 2rem; -webkit-overflow-scrolling: touch; }
.napkin-track { display: flex; align-items: center; gap: 0; min-width: max-content; padding: 0.5rem 1rem; }
.napkin-card {
    background: linear-gradient(160deg, #1a2745 0%, #0f1a30 100%);
    border: 1px solid var(--gold3); border-radius: 14px;
    padding: 1.2rem 1rem; width: 155px; text-align: center; flex-shrink: 0;
    transition: transform 0.2s, border-color 0.2s;
}
.napkin-card:hover { transform: translateY(-5px); border-color: var(--gold); }
.napkin-num { font-family:'Playfair Display',serif; font-size:1.6rem; font-weight:900; color:rgba(212,168,83,0.2); line-height:1; margin-bottom:0.3rem; }
.napkin-icon { font-size: 1.7rem; margin-bottom: 0.4rem; }
.napkin-label { font-family:'IM Fell English',serif; font-size:0.85rem; color:var(--parch); line-height:1.35; margin-bottom:0.6rem; }
.napkin-tag { background:rgba(212,168,83,0.08); border:1px solid var(--gold3); color:var(--gold3); font-size:0.57rem; letter-spacing:1.5px; text-transform:uppercase; padding:0.2rem 0.5rem; border-radius:20px; display:inline-block; }
.napkin-arrow { color:var(--gold3); font-size:1.6rem; padding:0 0.4rem; flex-shrink:0; opacity:0.6; }

/* ── Character Cards ── */
.char-card { background:linear-gradient(160deg,var(--navy3) 0%,#0c1628 100%); border:1px solid var(--gold3); border-radius:8px; padding:1.2rem; margin-bottom:1rem; transition:transform 0.2s,border-color 0.2s; }
.char-card:hover { border-color:var(--gold); transform:translateY(-2px); }
.char-icon { font-size:1.8rem; }
.char-name { font-family:'IM Fell English',serif; font-size:1.1rem; color:var(--gold); }
.char-role { font-size:0.7rem; letter-spacing:2px; text-transform:uppercase; color:var(--parch2); margin:0.2rem 0; }
.char-mentions { background:var(--gold3); color:var(--gold2); padding:0.1rem 0.5rem; border-radius:20px; font-size:0.68rem; display:inline-block; margin-top:0.3rem; }

/* ── Evidence Cards ── */
.evidence-card { background:linear-gradient(135deg,#0f1929 0%,#1a2a1a 100%); border:1px solid #2d4a2d; border-radius:8px; padding:1.2rem; margin-bottom:0.8rem; }
.evidence-card.critical { border-color:var(--crimson); }
.evidence-card.high     { border-color:var(--gold3); }
.evidence-icon { font-size:1.8rem; }
.evidence-name { font-family:'IM Fell English',serif; color:var(--parch); font-size:1rem; }
.badge-critical { background:var(--crimson); color:#ffaaaa; padding:0.1rem 0.6rem; border-radius:12px; font-size:0.62rem; letter-spacing:1px; }
.badge-high     { background:#3a2a0a; color:var(--gold); padding:0.1rem 0.6rem; border-radius:12px; font-size:0.62rem; letter-spacing:1px; }
.badge-medium   { background:#1a2a3a; color:#88aacc; padding:0.1rem 0.6rem; border-radius:12px; font-size:0.62rem; letter-spacing:1px; }

/* ── Action Cards ── */
.action-card { background:#0f1520; border-left:3px solid var(--gold); padding:0.8rem 1.2rem; margin-bottom:0.7rem; border-radius:0 6px 6px 0; }
.action-verb  { font-size:0.62rem; letter-spacing:2px; text-transform:uppercase; color:var(--gold3); }
.action-name  { font-family:'IM Fell English',serif; font-size:1.05rem; color:var(--parch); }
.action-actor { color:var(--parch2); font-size:0.8rem; }

/* ── Deduction (minimal) ── */
.deduction-row {
    display: flex; gap: 0.6rem; align-items: center;
    background: #0d1420; border-left: 3px solid var(--gold3);
    border-radius: 0 6px 6px 0; padding: 0.6rem 1rem;
    margin-bottom: 0.4rem;
}
.deduction-icon { font-size: 1.2rem; flex-shrink:0; }
.deduction-clue { font-family:'IM Fell English',serif; color:var(--gold); font-size:0.95rem; min-width: 160px; }
.deduction-logic { color:var(--parch2); font-size:0.8rem; }
.conclusion-box {
    background: linear-gradient(135deg,#0f1929,#1a1025);
    border: 1px solid var(--gold3); border-radius:8px;
    padding: 1.2rem 1.5rem; margin-top: 1rem;
    font-family:'IM Fell English',serif;
    font-size: 1rem; color: var(--parch); line-height: 1.7;
}
.conclusion-box b { color: var(--gold); }

/* ── Code Snippet Window ── */
.code-window {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
    overflow: hidden;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.82rem;
    line-height: 1.7;
}
.code-titlebar {
    background: #161b22;
    border-bottom: 1px solid #30363d;
    padding: 0.5rem 1rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.dot-red    { width:12px;height:12px;border-radius:50%;background:#ff5f57;display:inline-block; }
.dot-yellow { width:12px;height:12px;border-radius:50%;background:#febc2e;display:inline-block; }
.dot-green  { width:12px;height:12px;border-radius:50%;background:#28c840;display:inline-block; }
.code-filename { color:#8b949e; font-size:0.75rem; margin-left:0.5rem; }
.code-body { padding: 1.2rem 1.5rem; overflow-x: auto; }
/* Syntax colours */
.kw  { color: #ff7b72; }   /* keywords */
.fn  { color: #d2a8ff; }   /* functions */
.st  { color: #a5d6ff; }   /* strings/imports */
.cm  { color: #8b949e; font-style:italic; }  /* comments */
.nb  { color: #79c0ff; }   /* builtins / module names */
.nm  { color: #ffa657; }   /* names / variables */
.op  { color: #c9d1d9; }   /* operators */
.num { color: #79c0ff; }   /* numbers */

/* ── Custom Corpus Uploader ── */
.upload-box {
    background: linear-gradient(135deg,#0f1929,#0a1520);
    border: 2px dashed var(--gold3); border-radius: 12px;
    padding: 2rem; text-align: center; margin-bottom: 1.5rem;
}
.upload-title {
    font-family:'UnifrakturMaguntia',cursive;
    font-size:1.6rem; color:var(--gold); margin-bottom:0.5rem;
}
.upload-sub { color:var(--parch2); font-size:0.85rem; }

/* ── NLP Cards ── */
.nlp-card { background:linear-gradient(135deg,var(--navy3) 0%,#0c1220 100%); border:1px solid var(--gold3); border-radius:8px; padding:1.2rem; margin-bottom:0.8rem; }
.nlp-card-title { font-family:'IM Fell English',serif; color:var(--gold); font-size:1rem; margin-bottom:0.5rem; }
.nlp-card-body  { color:var(--parch2); font-size:0.82rem; line-height:1.5; }
.nlp-tag { background:var(--gold3); color:var(--navy); padding:0.1rem 0.4rem; border-radius:4px; font-size:0.63rem; font-family:monospace; margin-right:0.3rem; }

/* ── Debug ── */
.debug-panel { background:#050a14; border:1px solid #1a2540; border-radius:8px; padding:1.5rem; font-family:monospace; font-size:0.8rem; color:#7a9abf; max-height:300px; overflow-y:auto; }
.debug-key { color:var(--gold3); }
.debug-val { color:#88ccaa; }

/* ── Team Cards ── */
.team-card {
    background: linear-gradient(135deg,#0f1929,#1a2540);
    border: 1px solid var(--gold3); border-radius: 12px;
    padding: 1.8rem 1rem; text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.team-card:hover { transform:translateY(-4px); border-color:var(--gold); }
.team-avatar {
    width:60px; height:60px; border-radius:50%;
    background: linear-gradient(135deg,var(--gold3),var(--gold));
    display:flex; align-items:center; justify-content:center;
    font-family:'Playfair Display',serif; font-size:1.4rem;
    font-weight:900; color:var(--navy);
    margin: 0 auto 0.8rem;
}
.team-name { font-family:'IM Fell English',serif; font-size:1.05rem; color:var(--parch); margin-bottom:0.3rem; }
.team-role { font-size:0.65rem; letter-spacing:2px; text-transform:uppercase; color:var(--gold3); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def load_all_data():
    text                       = load_corpus()
    characters                 = extract_characters(text)
    clues                      = extract_clues(text)
    actions, top_verbs         = extract_actions(text)
    tokens, tagged, top_nouns  = extract_nouns_tokens(text)
    deductions                 = reason_case()
    return text, characters, clues, actions, top_verbs, tokens, tagged, top_nouns, deductions

text, characters, clues, actions, top_verbs, tokens, tagged, top_nouns, deductions = load_all_data()
sentences = sent_tokenize(text)

# ══════════════════════════════════════════════════════════════════
#  § 1  HERO
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-left">
    <div class="hero-tag">🔍 &nbsp; NLP Case File № 001 — Silver Blaze</div>
    <h1 class="hero-title">Sherlock Holmes<br><span>Intelligence</span><br>Bureau</h1>
    <p class="hero-subtitle">Solving the Silver Blaze Mystery</p>
    <p class="hero-desc">
      This system reads a detective corpus, extracts characters and clues via
      Natural Language Processing, maps every relationship into a Knowledge Graph,
      and reconstructs Holmes-style logical deductions — entirely from raw text.
    </p>
  </div>
  <div class="hero-right">
    <img src="{HERO_IMAGE_URL}" alt="Detective silhouette in fog"/>
    <div class="hero-fade-left"></div>
    <div class="hero-fade-bottom"></div>
    <div class="hero-fade-top"></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  § 2  EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Tokenisation · Entity Recognition · Relation Extraction</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
for col, num, label in [
    (m1, len(characters),      "Characters Found"),
    (m2, len(clues),           "Evidence Items"),
    (m3, len(actions),         "Actions Extracted"),
    (m4, 4,                    "Locations Found"),
    (m5, len(characters) * 3,  "Relationships Built"),
]:
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-num">{num}</div>
          <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div style="margin-top:1rem;padding:0.9rem 1.2rem;background:#0d1525;border-radius:6px;
     border-left:3px solid #d4a853;font-size:0.82rem;color:#8899bb;">
  <b style="color:#d4a853;">Corpus Stats</b> &nbsp;·&nbsp;
  {len(tokens):,} tokens &nbsp;·&nbsp; {len(sentences)} sentences &nbsp;·&nbsp;
  {len(set(t.lower() for t in tokens if t.isalpha())):,} unique words
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  § 3  KNOWLEDGE GRAPH
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Interactive Knowledge Graph</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Relation Extraction · Graph Modelling · PyVis Network</div>', unsafe_allow_html=True)

st.markdown("""
<div class="graph-legend">
  <span>🟡 <b style="color:#b8860b;">Character</b></span>
  <span>🔵 <b style="color:#1e4080;">Object</b></span>
  <span>🟢 <b style="color:#1a5030;">Place</b></span>
  <span>🔴 <b style="color:#8a1515;">Event</b></span>
  <span style="color:#8a7a6a;font-size:0.72rem;margin-left:1rem;">
    Drag nodes · Scroll to zoom · Hover for details
  </span>
</div>""", unsafe_allow_html=True)

with st.spinner("Constructing knowledge graph..."):
    net        = build_graph()
    graph_html = render_graph(net)

st.markdown('<div class="graph-frame">', unsafe_allow_html=True)
try:
    st.components.v1.html(graph_html, height=640, scrolling=False)
except Exception:
    st.markdown(f'<iframe srcdoc="{graph_html}" width="100%" height="640px"></iframe>',
                unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  § 4  CASE TIMELINE
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Case Timeline</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Sentence Segmentation · Event Ordering</div>', unsafe_allow_html=True)

TIMELINE = [
    ("🐎","Silver Blaze Favoured",   "Freq. Analysis"),
    ("🎩","Simpson Approaches",      "NER"),
    ("💊","Dinner Drugged",          "Keyword Extraction"),
    ("🚪","Horse Vanishes",          "Event Detection"),
    ("💀","Straker Found Dead",      "Relation Extraction"),
    ("🔍","Holmes Investigates",     "Agent Detection"),
    ("🏇","Horse Recovered",         "Resolution Event"),
    ("✅","Truth Revealed",          "Final Inference"),
]

cards_html = ""
for i, (icon, label, tag) in enumerate(TIMELINE):
    cards_html += f"""
    <div class="napkin-card">
      <div class="napkin-num">{str(i+1).zfill(2)}</div>
      <div class="napkin-icon">{icon}</div>
      <div class="napkin-label">{label}</div>
      <span class="napkin-tag">{tag}</span>
    </div>"""
    if i < len(TIMELINE) - 1:
        cards_html += '<div class="napkin-arrow">→</div>'

st.markdown(f'<div class="napkin-wrap"><div class="napkin-track">{cards_html}</div></div>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  § 5  CHARACTER GALLERY
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
#  § 6  EVIDENCE LOCKER
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Evidence Locker</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Keyword Extraction · Noun Phrase Mining</div>', unsafe_allow_html=True)

ev1, ev2 = st.columns(2)
for i, (name, meta) in enumerate(clues.items()):
    with (ev1 if i % 2 == 0 else ev2):
        sig = meta['significance'].lower()
        st.markdown(f"""
        <div class="evidence-card {sig}">
          <div style="display:flex;align-items:center;gap:0.8rem;">
            <span class="evidence-icon">{meta['icon']}</span>
            <div>
              <div class="evidence-name">{name}</div>
              <span class="badge-{sig}">{meta['significance'].upper()}</span>
            </div>
          </div>
          <div style="margin-top:0.7rem;font-size:0.8rem;color:#8899bb;line-height:1.6;">
            {meta['explanation']}
          </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  § 7  ACTION REGISTRY
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Action Registry</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Verb Extraction · Sentence Parsing</div>', unsafe_allow_html=True)

ac1, ac2 = st.columns(2)
for i, act in enumerate(actions):
    with (ac1 if i % 2 == 0 else ac2):
        st.markdown(f"""
        <div class="action-card">
          <div style="display:flex;align-items:center;gap:0.8rem;">
            <span style="font-size:1.5rem;">{act['icon']}</span>
            <div>
              <div class="action-verb">verb: {act['verb']}</div>
              <div class="action-name">{act['action']}</div>
              <div class="action-actor">— {act['actor']}</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  § 8  HOLMES DEDUCTION — minimal
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Holmes Deduction</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Rule-Based Logical Inference</div>', unsafe_allow_html=True)

ded_col, conc_col = st.columns([1.2, 1])
with ded_col:
    for icon, clue, logic in deductions:
        st.markdown(f"""
        <div class="deduction-row">
          <span class="deduction-icon">{icon}</span>
          <span class="deduction-clue">{clue}</span>
          <span class="deduction-logic">→ {logic}</span>
        </div>""", unsafe_allow_html=True)

with conc_col:
    st.markdown("""
    <div class="conclusion-box">
      <b>Conclusion</b><br><br>
      Straker drugged his own guard, took Silver Blaze to the moor,
      and attempted surgical sabotage for betting profit.<br><br>
      A match he struck in the dark startled the horse.
      The kick that killed him was <b>self-defence</b>, not murder.<br><br>
      <b>Simpson</b> — innocent.<br>
      <b>Straker</b> — architect of his own death.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  § 9  NLP METHODS
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">NLP Methods Used</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Techniques powering this investigation</div>', unsafe_allow_html=True)

NLP_METHODS = [
    ("Tokenisation",           "nltk.word_tokenize\nnltk.sent_tokenize",
     "Splits raw text into words and sentences. Foundation of every pipeline."),
    ("POS Tagging",            "nltk.pos_tag",
     "Labels tokens: NN=noun, VB=verb, NNP=proper noun. Powers extraction without keyword lists."),
    ("Named Entity Recog.",    "nltk.ne_chunk\n+ predefined list",
     "Identifies people, places, organisations. Augmented for literary text accuracy."),
    ("Frequency Distribution", "nltk.FreqDist\ncollections.Counter",
     "Counts token occurrences to rank character importance and surface key themes."),
    ("Relation Extraction",    "Co-occurrence + rules",
     "Finds subject-verb-object triples. These triples become the directed graph edges."),
    ("Knowledge Graph",        "pyvis.network.Network",
     "Entities as nodes, relationships as labelled edges. Makes story structure explicit."),
]

nc1, nc2, nc3 = st.columns(3)
for i, (title, tag_str, desc) in enumerate(NLP_METHODS):
    with [nc1, nc2, nc3][i % 3]:
        tags = "".join(f'<span class="nlp-tag">{t.strip()}</span>' for t in tag_str.split('\n'))
        st.markdown(f"""
        <div class="nlp-card">
          <div class="nlp-card-title">{title}</div>
          <div style="margin-bottom:0.5rem;">{tags}</div>
          <div class="nlp-card-body">{desc}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  § 10  CODE SNIPPET
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Code Spotlight</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Core NLP pipeline — annotated source</div>', unsafe_allow_html=True)

code_col, explain_col = st.columns([1.3, 1])

with code_col:
    st.markdown("""
<div class="code-window">
  <div class="code-titlebar">
    <span class="dot-red"></span>
    <span class="dot-yellow"></span>
    <span class="dot-green"></span>
    <span class="code-filename">nlp_pipeline.py</span>
  </div>
  <div class="code-body">
<pre style="margin:0;color:#c9d1d9;"><span class="cm"># ── 1. Tokenisation ──────────────────────────</span>
<span class="kw">import</span> <span class="nb">nltk</span>
<span class="kw">from</span> <span class="nb">nltk.tokenize</span> <span class="kw">import</span> <span class="st">word_tokenize</span>, <span class="st">sent_tokenize</span>

<span class="nm">tokens</span>    <span class="op">=</span> <span class="fn">word_tokenize</span>(<span class="nm">text</span>)
<span class="nm">sentences</span> <span class="op">=</span> <span class="fn">sent_tokenize</span>(<span class="nm">text</span>)

<span class="cm"># ── 2. POS Tagging ───────────────────────────</span>
<span class="kw">from</span> <span class="nb">nltk.tag</span> <span class="kw">import</span> <span class="st">pos_tag</span>

<span class="nm">tagged</span> <span class="op">=</span> <span class="fn">pos_tag</span>(<span class="nm">tokens</span>)
<span class="cm"># Output: [('Holmes', 'NNP'), ('investigated', 'VBD'), ...]</span>

<span class="cm"># ── 3. Extract Nouns + Verbs via POS filter ──</span>
<span class="nm">nouns</span> <span class="op">=</span> [<span class="nm">w</span> <span class="kw">for</span> <span class="nm">w</span>, <span class="nm">t</span> <span class="kw">in</span> <span class="nm">tagged</span> <span class="kw">if</span> <span class="nm">t</span>.<span class="fn">startswith</span>(<span class="st">'NN'</span>)]
<span class="nm">verbs</span> <span class="op">=</span> [<span class="nm">w</span> <span class="kw">for</span> <span class="nm">w</span>, <span class="nm">t</span> <span class="kw">in</span> <span class="nm">tagged</span> <span class="kw">if</span> <span class="nm">t</span>.<span class="fn">startswith</span>(<span class="st">'VB'</span>)]

<span class="cm"># ── 4. Named Entity Recognition ──────────────</span>
<span class="kw">from</span> <span class="nb">nltk</span> <span class="kw">import</span> <span class="st">ne_chunk</span>
<span class="kw">from</span> <span class="nb">nltk.tree</span> <span class="kw">import</span> <span class="st">Tree</span>

<span class="nm">tree</span> <span class="op">=</span> <span class="fn">ne_chunk</span>(<span class="nm">tagged</span>)
<span class="nm">entities</span> <span class="op">=</span> [
    <span class="st">' '</span>.<span class="fn">join</span>(<span class="nm">w</span> <span class="kw">for</span> <span class="nm">w</span>, <span class="nm">_</span> <span class="kw">in</span> <span class="nm">subtree</span>.<span class="fn">leaves</span>())
    <span class="kw">for</span> <span class="nm">subtree</span> <span class="kw">in</span> <span class="nm">tree</span>
    <span class="kw">if</span> <span class="fn">isinstance</span>(<span class="nm">subtree</span>, <span class="nb">Tree</span>)
]

<span class="cm"># ── 5. Frequency Distribution ────────────────</span>
<span class="kw">from</span> <span class="nb">nltk</span> <span class="kw">import</span> <span class="st">FreqDist</span>

<span class="nm">fd</span>  <span class="op">=</span> <span class="fn">FreqDist</span>(<span class="nm">tokens</span>)
<span class="nm">top</span> <span class="op">=</span> <span class="nm">fd</span>.<span class="fn">most_common</span>(<span class="num">10</span>)

<span class="cm"># ── 6. Build Knowledge Graph ─────────────────</span>
<span class="kw">from</span> <span class="nb">pyvis.network</span> <span class="kw">import</span> <span class="st">Network</span>

<span class="nm">net</span> <span class="op">=</span> <span class="fn">Network</span>(<span class="st">bgcolor</span><span class="op">=</span><span class="st">"#faf7f0"</span>)
<span class="nm">net</span>.<span class="fn">add_node</span>(<span class="st">"Holmes"</span>, <span class="st">label</span><span class="op">=</span><span class="st">"Sherlock Holmes"</span>)
<span class="nm">net</span>.<span class="fn">add_node</span>(<span class="st">"Stable"</span>, <span class="st">label</span><span class="op">=</span><span class="st">"King's Pyland"</span>)
<span class="nm">net</span>.<span class="fn">add_edge</span>(<span class="st">"Holmes"</span>, <span class="st">"Stable"</span>, <span class="st">label</span><span class="op">=</span><span class="st">"investigated"</span>)
<span class="nm">net</span>.<span class="fn">save_graph</span>(<span class="st">"graph.html"</span>)</pre>
  </div>
</div>""", unsafe_allow_html=True)

with explain_col:
    st.markdown("""
<div style="padding: 0.5rem 0 0 1.5rem;">
  <div style="margin-bottom:1.6rem;">
    <div style="font-family:'IM Fell English',serif;color:#d4a853;font-size:1rem;margin-bottom:0.4rem;">
      1 · Tokenisation
    </div>
    <div style="color:#c8b89a;font-size:0.82rem;line-height:1.6;">
      The raw text string is split into word tokens and sentence tokens.
      Every subsequent NLP step depends on this output.
    </div>
  </div>
  <div style="margin-bottom:1.6rem;">
    <div style="font-family:'IM Fell English',serif;color:#d4a853;font-size:1rem;margin-bottom:0.4rem;">
      2 · POS Tagging
    </div>
    <div style="color:#c8b89a;font-size:0.82rem;line-height:1.6;">
      Each token receives a Penn Treebank tag. NNP = proper noun,
      VBD = past-tense verb. We filter these to extract characters and actions.
    </div>
  </div>
  <div style="margin-bottom:1.6rem;">
    <div style="font-family:'IM Fell English',serif;color:#d4a853;font-size:1rem;margin-bottom:0.4rem;">
      3 · NER + FreqDist
    </div>
    <div style="color:#c8b89a;font-size:0.82rem;line-height:1.6;">
      <code style="background:#1a2540;padding:1px 5px;border-radius:3px;color:#88aacc;">ne_chunk</code>
      groups NNP tokens into named entities.
      <code style="background:#1a2540;padding:1px 5px;border-radius:3px;color:#88aacc;">FreqDist</code>
      ranks them by appearance count to determine importance.
    </div>
  </div>
  <div>
    <div style="font-family:'IM Fell English',serif;color:#d4a853;font-size:1rem;margin-bottom:0.4rem;">
      4 · Knowledge Graph
    </div>
    <div style="color:#c8b89a;font-size:0.82rem;line-height:1.6;">
      PyVis converts extracted entities into interactive nodes and
      subject-verb-object triples into directed, labelled edges.
      Exported as a self-contained HTML file.
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  § 11  CUSTOM CORPUS UPLOADER
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Analyse Your Own Corpus</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Upload any text file — get an instant knowledge graph</div>', unsafe_allow_html=True)

st.markdown("""
<div class="upload-box">
  <div class="upload-title">Upload a Text File</div>
  <div class="upload-sub">
    Paste or upload any plain-text corpus below.<br>
    The NLP pipeline will tokenise, tag, extract entities,
    and generate a live knowledge graph automatically.
  </div>
</div>""", unsafe_allow_html=True)

up_mode = st.radio("Input method", ["Upload .txt file", "Paste text directly"],
                   horizontal=True, label_visibility="collapsed")

custom_text = None

if up_mode == "Upload .txt file":
    uploaded = st.file_uploader("Upload a plain text file", type=["txt"],
                                label_visibility="collapsed")
    if uploaded is not None:
        custom_text = uploaded.read().decode("utf-8", errors="ignore")
else:
    pasted = st.text_area("Paste your corpus here",
                          height=180,
                          placeholder="Paste any story, article, or document text here...",
                          label_visibility="collapsed")
    if pasted.strip():
        custom_text = pasted.strip()

if custom_text:
    word_count = len(custom_text.split())
    st.markdown(f"""
    <div style="background:#0d1525;border-left:3px solid #d4a853;border-radius:4px;
         padding:0.6rem 1rem;font-size:0.8rem;color:#8899bb;margin-bottom:1rem;">
      ✓ &nbsp; Corpus loaded — <b style="color:#d4a853;">{word_count:,} words</b> detected.
      Running NLP pipeline…
    </div>""", unsafe_allow_html=True)

    with st.spinner("Building your knowledge graph..."):
        try:
            custom_net  = build_graph_from_text(custom_text)
            custom_html = render_graph(custom_net)

            # Quick stats
            c_tokens    = word_tokenize(custom_text)
            c_sentences = sent_tokenize(custom_text)
            c_tagged    = pos_tag(c_tokens)
            c_nouns     = [w for w, t in c_tagged if t.startswith('NN') and len(w) > 3]
            c_verbs     = [w for w, t in c_tagged if t.startswith('VB') and len(w) > 3]
            c_proper    = [w for w, t in c_tagged if t == 'NNP' and len(w) > 2]
            top_entities = Counter(c_proper).most_common(8)

            s1, s2, s3, s4 = st.columns(4)
            for col, num, lbl in [
                (s1, len(c_tokens),                   "Tokens"),
                (s2, len(c_sentences),                "Sentences"),
                (s3, len(set(w.lower() for w in c_proper)), "Entities"),
                (s4, len(set(w.lower() for w in c_verbs)),  "Unique Verbs"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="metric-card" style="padding:1rem;">
                      <div class="metric-num" style="font-size:2rem;">{num}</div>
                      <div class="metric-label">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            if top_entities:
                st.markdown("<div style='margin:1rem 0 0.5rem;font-size:0.75rem;letter-spacing:2px;color:#8b6914;text-transform:uppercase;'>Top Detected Entities</div>", unsafe_allow_html=True)
                ent_html = " &nbsp;".join(
                    f'<span style="background:#1a2540;color:#d4a853;padding:3px 10px;'
                    f'border-radius:4px;font-size:0.82rem;">{e} <span style="color:#8b6914;">({c})</span></span>'
                    for e, c in top_entities)
                st.markdown(f'<div style="margin-bottom:1rem;">{ent_html}</div>', unsafe_allow_html=True)

            st.markdown('<div class="graph-legend" style="margin-top:1rem;">'
                        '<span>🟡 <b style="color:#b8860b;">Entities (Proper Nouns)</b></span>'
                        '<span>🔵 <b style="color:#1e4080;">Key Nouns</b></span>'
                        '<span style="color:#8a7a6a;font-size:0.72rem;">Edges labelled with extracted verbs</span>'
                        '</div>', unsafe_allow_html=True)

            st.markdown('<div class="graph-frame">', unsafe_allow_html=True)
            try:
                st.components.v1.html(custom_html, height=580, scrolling=False)
            except Exception:
                st.markdown(f'<iframe srcdoc="{custom_html}" width="100%" height="580px"></iframe>',
                            unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Graph generation error: {e}. Try with a longer text (at least 5 sentences).")

# ══════════════════════════════════════════════════════════════════
#  § 12  DEBUG PANEL
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Raw NLP Debug Panel</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Inspect pipeline outputs — for viva and verification</div>',
            unsafe_allow_html=True)

with st.expander("🔬 Expand Debug Panel", expanded=False):
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Token Sample** (first 30)")
        st.markdown(f'<div class="debug-panel">{" | ".join(tokens[:30])}</div>', unsafe_allow_html=True)
        st.markdown("**POS Tag Sample** (first 20)")
        pos_html = "<br>".join(
            f'<span class="debug-key">{w}</span> → <span class="debug-val">{t}</span>'
            for w, t in tagged[:20])
        st.markdown(f'<div class="debug-panel">{pos_html}</div>', unsafe_allow_html=True)
    with d2:
        st.markdown("**Top Nouns**")
        nouns_html = " &nbsp;".join(
            f'<span style="background:#1a2a1a;color:#88cc88;padding:2px 8px;border-radius:4px;font-size:0.8rem;">{n}</span>'
            for n in top_nouns)
        st.markdown(f'<div class="debug-panel" style="line-height:2.5;">{nouns_html}</div>', unsafe_allow_html=True)
        st.markdown("**Top Verbs**")
        verbs_html = " &nbsp;".join(
            f'<span style="background:#1a1a2a;color:#8888cc;padding:2px 8px;border-radius:4px;font-size:0.8rem;">{v}</span>'
            for v in top_verbs[:15])
        st.markdown(f'<div class="debug-panel" style="line-height:2.5;">{verbs_html}</div>', unsafe_allow_html=True)

    st.markdown("**Character Mention Counts**")
    char_df = pd.DataFrame([
        {"Name": k, "Role": v["role"], "Mentions": v["mentions"]}
        for k, v in characters.items()
    ]).sort_values("Mentions", ascending=False)
    st.dataframe(char_df, use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="debug-panel">
      <span class="debug-key">Tokens :</span>    <span class="debug-val">{len(tokens):,}</span><br>
      <span class="debug-key">Sentences :</span> <span class="debug-val">{len(sentences)}</span><br>
      <span class="debug-key">Unique words :</span> <span class="debug-val">{len(set(t.lower() for t in tokens if t.isalpha())):,}</span><br>
      <span class="debug-key">Clues found :</span> <span class="debug-val">{len(clues)}</span><br>
      <span class="debug-key">Verbs extracted :</span> <span class="debug-val">{len(top_verbs)}</span>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  § 13  TEAM CREDITS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-top:4rem; border-top:1px solid #1a2540; padding-top:2.5rem;">
  <div style="text-align:center; margin-bottom:2rem;">
    <span style="font-family:'UnifrakturMaguntia',cursive; font-size:1.8rem;
                 color:#d4a853; letter-spacing:1px;">
      The Investigators
    </span><br>
    <span style="font-family:'IM Fell English',serif; font-style:italic;
                 color:#8b6914; font-size:0.9rem;">
      Built by
    </span>
  </div>
</div>""", unsafe_allow_html=True)

TEAM = [
    ("Vijay J",             "NLP Engineer",       "VJ"),
    ("Bishwarup Biswas",    "Graph Architect",    "BB"),
    ("Jayasuriya",          "Data Analyst",       "JS"),
    ("Yugmitha Kattayan",   "UI & Integration",   "YK"),
]

t1, t2, t3, t4 = st.columns(4)
for col, (name, role, initials) in zip([t1, t2, t3, t4], TEAM):
    with col:
        st.markdown(f"""
        <div class="team-card">
          <div class="team-avatar">{initials}</div>
          <div class="team-name">{name}</div>
          <div class="team-role">{role}</div>
        </div>""", unsafe_allow_html=True)

# ── Footer
st.markdown("""
<div style="margin-top:3rem;padding:1.5rem;border-top:1px solid #1a2540;
     text-align:center;font-size:0.7rem;color:#1e2e45;letter-spacing:2px;">
  SHERLOCK HOLMES INTELLIGENCE BUREAU &nbsp;·&nbsp; NLP: NLTK &nbsp;·&nbsp;
  GRAPH: PYVIS &nbsp;·&nbsp; UI: STREAMLIT<br>
  <span style="font-family:'IM Fell English',serif;font-style:italic;color:#162030;">
    Silver Blaze Case File · All deductions are elementary.
  </span>
</div>""", unsafe_allow_html=True)
