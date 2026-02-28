import gradio as gr
import json
import os
from urllib.parse import quote
from retrieval.retriever import Retriever
from llm.llm_service import LLMService
from llm.query_rewriter import QueryRewriter
import whisper
import torch
from storage.conversation_store import (
    create_conversation,
    save_conversation,
    load_conversation,
    list_conversations,
)

# ----------------------------
# Core Components
# ----------------------------

retriever = Retriever(similarity_threshold=0.35, top_k=8)
llm_service = LLMService(model="mistral")
rewriter = QueryRewriter(model="mistral")

# ----------------------------
# Load Whisper model
# ----------------------------

device = "mps" if torch.backends.mps.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=device)

# ----------------------------
# Voice → Text
# ----------------------------

def transcribe_audio(audio_path):
    if audio_path is None:
        return ""
    result = whisper_model.transcribe(audio_path)
    return result["text"].strip()

# ----------------------------
# Chat Logic
# ----------------------------

def load_conversations_on_start():
    return gr.update(
        choices=[
            (title, convo_id)
            for title, convo_id in list_conversations()
        ]
    )

def generate_response(message, history, selected_book, convo_id):

    if history is None:
        history = []

    new_convo_created = False

    if not convo_id:
        title = message.strip()[:60]
        convo_id = create_conversation(title)
        new_convo_created = True

    rewritten_query = rewriter.rewrite(message, history)

    history.append({"role": "user", "content": message})

    if selected_book == "All Books":
        retrieval_response = retriever.retrieve(rewritten_query)
    else:
        retrieval_response = retriever.retrieve(
            rewritten_query,
            book_id=selected_book
        )

    assistant_message = {"role": "assistant", "content": ""}
    history.append(assistant_message)

    for token in llm_service.generate_answer_stream(
        rewritten_query,
        retrieval_response
    ):
        assistant_message["content"] += token
        yield history, convo_id, None

    if retrieval_response["status"] == "success":
        citations = "\n\n---\n**Sources:**\n"
        collected_diagrams = []

        def _extract_hierarchy(raw_value):
            if not raw_value:
                return []
            if isinstance(raw_value, list):
                return [str(x) for x in raw_value]
            if isinstance(raw_value, str):
                candidate = raw_value.strip()
                if not candidate:
                    return []
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, list):
                        return [str(x) for x in parsed]
                except Exception:
                    pass
                return [candidate]
            return [str(raw_value)]

        def _extract_diagram_paths(raw_value):
            if not raw_value:
                return []
            if isinstance(raw_value, list):
                return [p for p in raw_value if isinstance(p, str) and p.strip()]
            if isinstance(raw_value, str):
                candidate = raw_value.strip()
                if not candidate:
                    return []
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, list):
                        return [p for p in parsed if isinstance(p, str) and p.strip()]
                except Exception:
                    pass
                if ";" in candidate:
                    return [p.strip() for p in candidate.split(";") if p.strip()]
                return [candidate]
            return []

        for result in retrieval_response["results"]:
            meta = result["metadata"]
            hierarchy = _extract_hierarchy(meta.get("hierarchy"))
            hierarchy_str = " → ".join(hierarchy)
            diagram_paths = _extract_diagram_paths(meta.get("diagram_paths"))

            for diagram in diagram_paths:
                if diagram not in collected_diagrams:
                    collected_diagrams.append(diagram)

            citations += (
                f"- {meta.get('book_name', 'Unknown Source')}\n"
                f"  {hierarchy_str}\n"
                f"  (Pages {meta.get('start_page')}–{meta.get('end_page')})\n\n"
            )

        if collected_diagrams:
            def _as_file_url(path: str) -> str:
                return f"/gradio_api/file={quote(path, safe='/')}"

            citations += "**Diagrams:**\n"
            for path in collected_diagrams[:12]:
                # Render clickable path plus inline preview in Gradio markdown.
                file_url = _as_file_url(path)
                citations += f"- [{path}]({file_url})\n"
            citations += "\n**Diagram Preview:**\n"
            for path in collected_diagrams[:4]:
                file_url = _as_file_url(path)
                citations += f"![diagram]({file_url})\n\n"
            citations += "\n"

        assistant_message["content"] += citations
        yield history, convo_id, None

    save_conversation(convo_id, history)

    # Refresh dropdown if new conversation created
    if new_convo_created:
        yield history, convo_id, gr.update(
            choices=[
                (title, cid)
                for title, cid in list_conversations()
            ],
            value=convo_id
        )
    else:
        yield history, convo_id, None

# ----------------------------
# UI
# ----------------------------

with gr.Blocks() as demo:

    gr.Markdown("# 🩺 AIRA - Anesthesia Resident Assistant")

    conversation_id = gr.State()

    with gr.Row():

        with gr.Column(scale=2):

            gr.Markdown("## 💬 Conversations")

            convo_list = gr.Dropdown(
                choices=[],
                label="Select Conversation",
                interactive=True
            )

        with gr.Column(scale=8):

            chatbot = gr.Chatbot(height=500)

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about anesthesia...",
                    scale=6
                )

                mic = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎤",
                    scale=2
                )

            book_selector = gr.Dropdown(
                choices=[
                    "All Books",
                    "miller_9e",
                    "barash-clinical-anaesthesiology-231220_124533"
                ],
                value="All Books",
                label="Select Book Scope"
            )

            with gr.Row():
                send_btn = gr.Button("Send")
                stop_btn = gr.Button("Stop")
                clear_btn = gr.Button("Clear")

    mic.change(
        fn=transcribe_audio,
        inputs=mic,
        outputs=msg
    )

    demo.load(
        fn=load_conversations_on_start,
        outputs=convo_list
    )

    def load_selected_chat(convo_id):
        history = load_conversation(convo_id)
        return history, convo_id

    convo_list.change(
        fn=load_selected_chat,
        inputs=convo_list,
        outputs=[chatbot, conversation_id]
    )

    send_event = msg.submit(
        generate_response,
        inputs=[msg, chatbot, book_selector, conversation_id],
        outputs=[chatbot, conversation_id, convo_list]
    )

    send_btn_event = send_btn.click(
        generate_response,
        inputs=[msg, chatbot, book_selector, conversation_id],
        outputs=[chatbot, conversation_id, convo_list]
    )

    stop_btn.click(fn=None, cancels=[send_event, send_btn_event])
    clear_btn.click(lambda: [], None, chatbot)

demo.queue()

demo.launch(
    share=True,
    theme=gr.themes.Soft(),
    allowed_paths=[
        os.path.abspath("data_bank/diagrams")
    ]
)
