import json
from datetime import datetime
from pathlib import Path
from config import CONV_DIR


def list_conversations():
    convos = sorted(Path(CONV_DIR).glob("*.json"), reverse=True)
    if not convos:
        print("  No saved conversations.")
        return None
    print("\nSaved conversations:")
    for i, c in enumerate(convos[:10], 1):
        try:
            data = json.loads(c.read_text())
            turns = len(data["history"])
            has_summary = "✦" if any("summary" in t for t in data["history"]) else " "
            print(f"  [{i}]{has_summary} {data['started']}  ({turns} turns)  — {c.stem}")
        except:
            pass
    return convos


def load_conversation():
    convos = list_conversations()
    if not convos:
        return [], None
    choice = input("\nLoad a conversation? Enter number or press Enter to start fresh: ").strip()
    if not choice:
        return [], None
    try:
        idx = int(choice) - 1
        data = json.loads(convos[idx].read_text())
        history = data["history"]
        turns = len(history)
        # find latest summary if any
        latest_summary = ""
        for t in reversed(history):
            if "summary" in t:
                latest_summary = t["summary"]
                break
        if latest_summary:
            print(f"  ✓ Loaded: {data['started']} ({turns} turns, summary available)")
        else:
            print(f"  ✓ Loaded: {data['started']} ({turns} turns)")
        return history, convos[idx]
    except:
        print("  Invalid choice, starting fresh.")
        return [], None


def save_conversation(history, filepath=None, model_name=""):
    if not history:
        return filepath
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = Path(CONV_DIR) / f"convo_{timestamp}.json"
    data = {
        "started": str(filepath.stem).replace("convo_", ""),
        "model": model_name,
        "history": history,
    }
    filepath.write_text(json.dumps(data, indent=2))
    return filepath


def format_history_for_prompt(history, max_turns=8):
    recent = history[-max_turns:]
    return "\n\n".join(
        f"User: {t['question']}\nAssistant: {t['answer']}"
        for t in recent
    ).strip()
