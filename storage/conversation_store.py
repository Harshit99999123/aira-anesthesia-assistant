import os
import json
import uuid

DATA_DIR = "chat_history"
os.makedirs(DATA_DIR, exist_ok=True)


def create_conversation(title):
    convo_id = str(uuid.uuid4())
    path = os.path.join(DATA_DIR, f"{convo_id}.json")

    data = {
        "title": title,
        "messages": []
    }

    with open(path, "w") as f:
        json.dump(data, f)

    return convo_id


def save_conversation(convo_id, history):
    path = os.path.join(DATA_DIR, f"{convo_id}.json")

    if not os.path.exists(path):
        return

    with open(path, "r") as f:
        data = json.load(f)

    data["messages"] = history

    with open(path, "w") as f:
        json.dump(data, f)


def load_conversation(convo_id):
    path = os.path.join(DATA_DIR, f"{convo_id}.json")

    if not os.path.exists(path):
        return []

    with open(path, "r") as f:
        data = json.load(f)

    return data.get("messages", [])


def list_conversations():
    conversations = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".json"):
            convo_id = file.replace(".json", "")
            path = os.path.join(DATA_DIR, file)

            with open(path, "r") as f:
                data = json.load(f)

            title = data.get("title", convo_id)
            conversations.append((title, convo_id))

    return conversations