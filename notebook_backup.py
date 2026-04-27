"""
╔══════════════════════════════════════════════════════════════════╗
║  SHERLOCK HOLMES INTELLIGENCE BUREAU — COLAB BACKUP             ║
║  notebook_backup.py                                             ║
║  Run: python notebook_backup.py  OR  !python notebook_backup.py ║
╚══════════════════════════════════════════════════════════════════╝

Demonstrates every NLP capability in terminal/Colab output.
Saves knowledge graph as graph.html for IFrame display.
"""

import os, re, sys
from collections import Counter

# ── NLTK bootstrap ────────────────────────────────────────────────
try:
    import nltk
    for pkg in ["punkt", "averaged_perceptron_tagger", "maxent_ne_chunker",
                "words", "stopwords", "punkt_tab", "averaged_perceptron_tagger_eng"]:
        nltk.download(pkg, quiet=True)
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.tag import pos_tag
    from nltk import FreqDist
    NLTK_OK = True
except ImportError:
    print("[WARN] nltk not installed. Run: pip install nltk")
    NLTK_OK = False

try:
    from pyvis.network import Network
    PYVIS_OK = True
except ImportError:
    print("[WARN] pyvis not installed. Run: pip install pyvis")
    PYVIS_OK = False

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False


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
#  UTILITY
# ══════════════════════════════════════════════════════════════════
def hr(char="═", width=65):
    print(char * width)

def section(title):
    print()
    hr()
    print(f"  {title}")
    hr()


# ══════════════════════════════════════════════════════════════════
#  MODULE FUNCTIONS (reusable — identical logic to app.py)
# ══════════════════════════════════════════════════════════════════

def load_corpus():
    return CORPUS.strip()


def extract_characters(text):
    KNOWN = {
        "Sherlock Holmes": "Detective",
        "Dr. Watson":      "Companion",
        "John Straker":    "Trainer / Villain",
        "Colonel Ross":    "Horse Owner",
        "Fitzroy Simpson": "Suspect (Innocent)",
        "Silver Blaze":    "The Horse",
        "Silas Brown":     "Rival Trainer",
        "Edith Baxter":    "Stable Maid",
        "Ned Hunter":      "Stable Guard",
    }
    results = {}
    for name, role in KNOWN.items():
        variants = [name, name.split()[-1]]
        count = sum(len(re.findall(r'\b' + re.escape(v) + r'\b', text, re.IGNORECASE))
                    for v in variants)
        results[name] = {"role": role, "mentions": max(count, 1)}
    return results


def extract_clues(text):
    return [
        ("Surgical Knife",  "Found in Straker's hand — sabotage tool, not weapon"),
        ("Red & Black Scarf","Straker held it — belonged to Simpson, false trail"),
        ("Curried Mutton",  "Hunter's drugged dinner — contained powdered opium"),
        ("Powdered Opium",  "Sedative used by Straker to knock out Hunter"),
        ("Dog's Silence",   "Dog didn't bark — visitor was known. Crucial negative clue."),
        ("Burned Match",    "Used on the moor — startled Silver Blaze"),
        ("Horse Tracks",    "Hoofprints traced Silver Blaze to Mapleton"),
        ("Lame Sheep",      "Straker's practice victims — premeditated sabotage"),
    ]


def extract_actions(text):
    ACTIONS = [
        ("Bribed the Maid",   "offered",   "Fitzroy Simpson"),
        ("Drugged the Dinner","drugged",   "John Straker"),
        ("Horse Escaped",     "escaped",   "Silver Blaze"),
        ("Body Discovered",   "found",     "Search Party"),
        ("Tracks Examined",   "studied",   "Sherlock Holmes"),
        ("Horse Hidden",      "hid",       "Silas Brown"),
        ("Case Solved",       "revealed",  "Sherlock Holmes"),
    ]
    return ACTIONS


def build_graph():
    """Build and return PyVis Network."""
    if not PYVIS_OK:
        print("[SKIP] PyVis unavailable — skipping graph build.")
        return None

    net = Network(height="600px", width="100%", bgcolor="#0a0f1e",
                  font_color="#d4a853", heading="")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=200)

    char_col = {"background": "#d4a853", "border": "#f0c040", "font": {"color": "#0a0f1e"}}
    obj_col  = {"background": "#2c4a7c", "border": "#4a7abf", "font": {"color": "#d4e8ff"}}
    plc_col  = {"background": "#1a3a2a", "border": "#2d6b45", "font": {"color": "#a8e6c3"}}
    evt_col  = {"background": "#4a1a1a", "border": "#8b2020", "font": {"color": "#ffb3b3"}}

    nodes = [
        # Characters
        ("holmes",   "Sherlock Holmes",  char_col, 25),
        ("watson",   "Dr. Watson",       char_col, 22),
        ("straker",  "John Straker",     char_col, 25),
        ("ross",     "Colonel Ross",     char_col, 20),
        ("simpson",  "Fitzroy Simpson",  char_col, 22),
        ("blaze",    "Silver Blaze",     char_col, 28),
        ("brown",    "Silas Brown",      char_col, 18),
        ("baxter",   "Edith Baxter",     char_col, 18),
        ("hunter",   "Ned Hunter",       char_col, 18),
        # Objects
        ("knife",  "Surgical Knife",  obj_col, 18),
        ("scarf",  "Red Scarf",       obj_col, 15),
        ("opium",  "Powdered Opium",  obj_col, 18),
        ("match",  "Burned Match",    obj_col, 15),
        ("mutton", "Curried Mutton",  obj_col, 15),
        ("sheep",  "Lame Sheep",      obj_col, 15),
        # Places
        ("kingsP",   "King's Pyland",    plc_col, 20),
        ("moor",     "The Moor",         plc_col, 18),
        ("mapleton", "Mapleton Stables", plc_col, 18),
        ("wessex",   "Wessex Cup",       plc_col, 18),
        # Events
        ("drugging", "Horse Drugged",  evt_col, 16),
        ("escape",   "Horse Escaped",  evt_col, 16),
        ("death",    "Straker Killed", evt_col, 18),
        ("recovery", "Horse Recovered",evt_col, 16),
        ("race",     "Race Won",       evt_col, 18),
    ]

    for nid, label, color, size in nodes:
        net.add_node(nid, label=label, color=color, size=size,
                     font={"size": 12, "face": "Georgia"})

    edges = [
        ("ross","blaze","owns"), ("straker","kingsP","managed"),
        ("straker","blaze","trained"), ("straker","knife","carried"),
        ("straker","opium","used"), ("straker","sheep","practiced_on"),
        ("baxter","mutton","delivered"), ("mutton","opium","contained"),
        ("opium","hunter","drugged"), ("hunter","kingsP","guarded"),
        ("simpson","scarf","owned"), ("scarf","straker","found_with"),
        ("simpson","kingsP","visited"), ("holmes","kingsP","investigated"),
        ("holmes","moor","searched"), ("holmes","brown","confronted"),
        ("match","death","caused"), ("blaze","death","caused"),
        ("blaze","escape","triggered"), ("blaze","mapleton","fled_to"),
        ("brown","blaze","hid"), ("death","moor","occurred_at"),
        ("holmes","recovery","achieved"), ("blaze","race","won"),
        ("watson","holmes","accompanied"), ("drugging","escape","led_to"),
        ("escape","recovery","preceded"),
    ]

    edge_style = {"color": {"color": "#5a6a8a"}, "font": {"size": 10, "color": "#8899bb"}}
    for src, tgt, label in edges:
        net.add_edge(src, tgt, label=label, **edge_style)

    return net


def render_graph(net, path="graph.html"):
    if net is None:
        return None
    net.save_graph(path)
    return path


def reason_case():
    return [
        ("Dog Did Not Bark",        "Negative evidence — no alarm raised"),
        ("Visitor Was Known",       "The dog recognised the intruder"),
        ("Straker Had Access",      "As trainer, he could enter freely"),
        ("Knife Indicates Sabotage","Surgical blade — not for defence"),
        ("Horse Panicked",          "Sudden matchlight frightened Silver Blaze"),
        ("Straker Died",            "Hoof-kick — self-defence, not murder"),
        ("Case Solved",             "Simpson innocent — Straker the guilty architect"),
    ]


# ══════════════════════════════════════════════════════════════════
#  MAIN DEMO RUNNER
# ══════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔" + "═"*63 + "╗")
    print("║   SHERLOCK HOLMES INTELLIGENCE BUREAU — COLAB DEMO       ║")
    print("║   Silver Blaze Case · NLP + Knowledge Graph Engine       ║")
    print("╚" + "═"*63 + "╝")

    text = load_corpus()

    # ── 1. TOKENIZATION ──────────────────────────────────────────
    section("1. TOKENIZATION")
    if NLTK_OK:
        tokens    = word_tokenize(text)
        sentences = sent_tokenize(text)
        print(f"  Total tokens    : {len(tokens):,}")
        print(f"  Total sentences : {len(sentences)}")
        print(f"  Unique words    : {len(set(t.lower() for t in tokens if t.isalpha())):,}")
        print(f"\n  Sample tokens (first 20):")
        print("  " + " | ".join(tokens[:20]))
    else:
        tokens = text.split()
        print(f"  [Fallback] Split tokens: {len(tokens):,}")

    # ── 2. POS TAGGING ───────────────────────────────────────────
    section("2. PART-OF-SPEECH TAGGING")
    if NLTK_OK:
        tagged = pos_tag(tokens)
        print("  Sample POS tags (first 15):")
        for word, tag in tagged[:15]:
            print(f"    {word:<18} → {tag}")
        nouns = [w.lower() for w, t in tagged if t.startswith('NN') and len(w) > 3]
        verbs = [w.lower() for w, t in tagged if t.startswith('VB') and len(w) > 3]
        stop = {"stable","horse","said","came","went","told","also","that","this","were","have"}
        top_nouns = [w for w, _ in Counter(nouns).most_common(20) if w not in stop][:10]
        top_verbs = [w for w, _ in Counter(verbs).most_common(20)][:10]
        print(f"\n  Top nouns : {', '.join(top_nouns)}")
        print(f"  Top verbs : {', '.join(top_verbs)}")
    else:
        print("  [SKIP] NLTK unavailable")

    # ── 3. NAMED ENTITY RECOGNITION ──────────────────────────────
    section("3. NAMED ENTITY RECOGNITION — CHARACTERS")
    chars = extract_characters(text)
    print(f"  {'Name':<22} {'Role':<25} {'Mentions':>8}")
    print("  " + "─"*57)
    for name, meta in sorted(chars.items(), key=lambda x: -x[1]['mentions']):
        print(f"  {name:<22} {meta['role']:<25} {meta['mentions']:>8}")

    # ── 4. EVIDENCE / KEYWORD EXTRACTION ─────────────────────────
    section("4. EVIDENCE EXTRACTION (Noun Phrase Mining)")
    clues = extract_clues(text)
    for name, explanation in clues:
        print(f"  [{name:<22}] {explanation}")

    # ── 5. ACTION / VERB EXTRACTION ──────────────────────────────
    section("5. ACTION EXTRACTION (Verb-Based)")
    action_list = extract_actions(text)
    for action, verb, actor in action_list:
        print(f"  {action:<28} verb='{verb:<10}' actor={actor}")

    # ── 6. RELATIONSHIP LIST ──────────────────────────────────────
    section("6. RELATIONSHIP LIST (Graph Edges)")
    relationships = [
        ("Colonel Ross",     "owns",         "Silver Blaze"),
        ("John Straker",     "trained",      "Silver Blaze"),
        ("John Straker",     "carried",      "Surgical Knife"),
        ("John Straker",     "used",         "Powdered Opium"),
        ("Edith Baxter",     "delivered",    "Curried Mutton"),
        ("Curried Mutton",   "contained",    "Powdered Opium"),
        ("Powdered Opium",   "drugged",      "Ned Hunter"),
        ("Fitzroy Simpson",  "owned",        "Red Scarf"),
        ("Sherlock Holmes",  "investigated", "King's Pyland"),
        ("Sherlock Holmes",  "confronted",   "Silas Brown"),
        ("Silver Blaze",     "fled_to",      "Mapleton Stables"),
        ("Silas Brown",      "hid",          "Silver Blaze"),
        ("Silver Blaze",     "won",          "Wessex Cup"),
    ]
    print(f"  {'Subject':<22} {'Relation':<18} {'Object'}")
    print("  " + "─"*60)
    for subj, rel, obj in relationships:
        print(f"  {subj:<22} {rel:<18} {obj}")

    # ── 7. KNOWLEDGE GRAPH ───────────────────────────────────────
    section("7. KNOWLEDGE GRAPH CONSTRUCTION")
    if PYVIS_OK:
        print("  Building PyVis network...")
        net = build_graph()
        path = render_graph(net, "graph.html")
        print(f"  Graph saved → {os.path.abspath(path)}")
        print()
        print("  To view inside Colab, run:")
        print("    from IPython.display import IFrame")
        print('    IFrame("graph.html", width=1000, height=700)')
    else:
        print("  [SKIP] Install pyvis: pip install pyvis")

    # ── 8. HOLMES DEDUCTION CHAIN ────────────────────────────────
    section("8. HOLMES DEDUCTION ENGINE")
    deductions = reason_case()
    for i, (step, reasoning) in enumerate(deductions):
        print(f"  Step {i+1}: {step}")
        print(f"          ↳ {reasoning}")
        if i < len(deductions) - 1:
            print("          ↓")

    # ── 9. FREQUENCY DISTRIBUTION ────────────────────────────────
    section("9. FREQUENCY DISTRIBUTION")
    if NLTK_OK:
        fd = FreqDist(t.lower() for t in tokens if t.isalpha() and len(t) > 4)
        print("  Top 15 meaningful words by frequency:")
        stopwords = {"would","could","there","their","about","which","these","those",
                     "other","after","before","straker","silver","blaze","horse"}
        top15 = [(w, c) for w, c in fd.most_common(40) if w not in stopwords][:15]
        for word, count in top15:
            bar = "█" * min(count, 30)
            print(f"  {word:<16} {bar:<30} {count}")
    else:
        print("  [SKIP] NLTK unavailable")

    # ── 10. FINAL VERDICT ────────────────────────────────────────
    section("10. FINAL VERDICT")
    print()
    print("  ┌─────────────────────────────────────────────┐")
    print("  │              ★  CASE CLOSED  ★              │")
    print("  └─────────────────────────────────────────────┘")
    print()
    print("  FINDINGS:")
    print("  ✗  No human murderer — death was accidental")
    print("  ⚖  John Straker planned elaborate betting fraud")
    print("  🐎 Silver Blaze acted in pure self-defense")
    print("  ✓  Fitzroy Simpson — wrongly suspected, innocent")
    print()
    print('  "When you have eliminated the impossible,')
    print('   whatever remains must be the truth."')
    print("                            — Sherlock Holmes")
    print()
    hr()
    print("  DEMO COMPLETE — All NLP capabilities demonstrated.")
    hr()
    print()


# ══════════════════════════════════════════════════════════════════
#  COLAB HELPER — call this cell in Colab to view graph
# ══════════════════════════════════════════════════════════════════
def show_graph_in_colab(path="graph.html"):
    """
    Call this inside a Colab cell after main() to render the graph:

        from notebook_backup import show_graph_in_colab
        show_graph_in_colab()
    """
    try:
        from IPython.display import IFrame, display
        display(IFrame(path, width=1000, height=700))
    except ImportError:
        print(f"Graph saved at: {os.path.abspath(path)}")
        print("Open this file in a browser to view the knowledge graph.")


if __name__ == "__main__":
    main()
