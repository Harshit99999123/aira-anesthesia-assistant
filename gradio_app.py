import gradio as gr
from retrieval.retriever import Retriever
from llm.llm_service import LLMService
from llm.query_rewriter import QueryRewriter
import whisper
import torch


retriever = Retriever(similarity_threshold=0.35, top_k=8)
llm_service = LLMService(model="mistral")
rewriter = QueryRewriter(model="mistral")

# ----------------------------
# Load Whisper model
# ----------------------------

device = "mps" if torch.backends.mps.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=device)


# ----------------------------
# Voice → Text (Auto-fill textbox)
# ----------------------------

def transcribe_audio(audio_path):
    if audio_path is None:
        return ""

    result = whisper_model.transcribe(audio_path)
    return result["text"].strip()


def generate_response(message, history, selected_book):

    rewritten_query = rewriter.rewrite(message, history)

    history.append({"role": "user", "content": message})

    # Scope control
    if selected_book == "All Books":
        retrieval_response = retriever.retrieve(rewritten_query)
    else:
        retrieval_response = retriever.retrieve(
            rewritten_query,
            book_id=selected_book
        )

    assistant_message = {"role": "assistant", "content": ""}
    history.append(assistant_message)

    # Stream tokens using rewritten query
    for token in llm_service.generate_answer_stream(
        rewritten_query,
        retrieval_response
    ):
        assistant_message["content"] += token
        yield history

    # Add citations if retrieval succeeded
    if retrieval_response["status"] == "success":

        citations = "\n\n---\n**Sources:**\n"

        for result in retrieval_response["results"]:
            meta = result["metadata"]

            hierarchy = meta.get("hierarchy", [])
            hierarchy_str = " → ".join(hierarchy)

            citations += (
                f"- {meta.get('book_name', 'Unknown Source')}\n"
                f"  {hierarchy_str}\n"
                f"  (Pages {meta.get('start_page')}–{meta.get('end_page')})\n\n"
            )

        assistant_message["content"] += citations
        yield history

# ----------------------------
# UI
# ----------------------------

with gr.Blocks() as demo:

    gr.Markdown("# 🩺 AIRA - Anesthesia Resident Assistant")

    chatbot = gr.Chatbot(height=500)

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask about anesthesia...",
            scale=8,
            container=False
        )

    book_selector = gr.Dropdown(
        choices=["All Books", "miller_9e", "barash-clinical-anaesthesiology-231220_124533"],
        value="All Books",
        label="Select Book Scope"
    )

    with gr.Row():
        send_btn = gr.Button("Send")
        stop_btn = gr.Button("Stop")
        clear_btn = gr.Button("Clear")

    # 🎤 When audio recording stops → auto-fill textbox
    mic = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="🎤",
        scale=1
    )
    mic.change(
        fn=transcribe_audio,
        inputs=mic,
        outputs=msg  # output goes directly into textbox
    )

    # Send message
    send_event = msg.submit(generate_response, [msg, chatbot, book_selector], chatbot)
    send_btn_event = send_btn.click(generate_response, [msg, chatbot, book_selector], chatbot)

    stop_btn.click(fn=None, cancels=[send_event, send_btn_event])
    clear_btn.click(lambda: [], None, chatbot)

demo.queue()
demo.launch(share=True, theme=gr.themes.Soft())
