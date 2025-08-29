import json
import time
from datetime import datetime
from collections import defaultdict, deque
from mistralai import Mistral
from mistralai.models import SDKError
import httpx
import os

os.environ['MISTRAL_API_KEY'] = 'cwBe93dZVZ6zGjWlNVS3CoQGkReXNCJI'
API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=API_KEY)


def get_mood_from_llm(message_text, retries=3, delay=2):
    """Ask Mistral AI to classify player mood based on the latest message."""
    prompt = f"""
You are analyzing a player's message in a fantasy RPG game.
Classify the player's emotional tone as one of the following:
["friendly", "angry", "neutral", "confused", "excited", "sad"].

Message: "{message_text}"

Return ONLY one word from the list above.
"""

    for attempt in range(retries):
        try:
            response = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
            )
            mood = response.choices[0].message.content.strip().lower()
            allowed = {"friendly", "angry", "neutral", "confused", "excited", "sad"}
            if mood not in allowed:
                mood = "neutral"
            return mood
        except (SDKError, httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
            print(f" Mood API failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)

    return "neutral" 

def generate_reply(player_id, mood, history, message, retries=3, delay=2):
    """Generate NPC reply using Mistral model with retry & fallback."""
    prompt = f"""
You are an NPC in a fantasy RPG game.
The player's current mood state is: {mood}.
Here are their last few messages:
{history}

The player just said: "{message}"

Respond in a short, in-character NPC reply.
Keep the tone consistent with the NPC's mood.
"""

    for attempt in range(retries):
        try:
            response = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except (SDKError, httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
            print(f" Reply API failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)

    try:
        response = client.chat.complete(
            model="mistral-tiny-latest",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(NPC stays silent due to error: {e})"


def main():
    with open("players.json", "r") as f:
        messages = json.load(f)

    messages.sort(key=lambda x: datetime.fromisoformat(x["timestamp"]))

    player_states = defaultdict(lambda: {
        "history": deque(maxlen=3),
        "mood": "neutral"
    })

    results = []

    for msg in messages:
        pid, text, ts = msg["player_id"], msg["text"], msg["timestamp"]

        state = player_states[pid]
        history = list(state["history"])

        new_mood = get_mood_from_llm(text)
        state["mood"] = new_mood

        reply = generate_reply(pid, new_mood, history, text)
        time.sleep(1)  

        state["history"].append(text)

        results.append({
            "player_id": pid,
            "timestamp": ts,
            "message": text,
            "npc_reply": reply,
            "history": history,
            "npc_mood": new_mood
        })

    with open("npc_results.json", "w") as out_f:
        json.dump(results, out_f, indent=4)

    print(" Results saved to npc_results.json")


if __name__ == "__main__":
    main()
