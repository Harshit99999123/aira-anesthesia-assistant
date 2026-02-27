import gradio as gr
from retrieval.retriever import Retriever
from llm.llm_service import LLMService
from llm.query_rewriter import QueryRewriter

retriever = Retriever(similarity_threshold=0.4, top_k=5)
llm_service = LLMService(model="mistral")
rewriter = QueryRewriter(model="mistral")


def generate_response(message, history):

    rewritten_query = rewriter.rewrite(message, history)

    history.append({"role": "user", "content": message})

    retrieval_response = retriever.retrieve(rewritten_query)

    assistant_message = {"role": "assistant", "content": ""}
    history.append(assistant_message)

    # Stream tokens
    for token in llm_service.generate_answer_stream(message, retrieval_response):
        assistant_message["content"] += token
        yield history

    # Add citations at end if success
    if retrieval_response["status"] == "success":
        citations = "\n\n---\n**Sources:**\n"
        for result in retrieval_response["results"]:
            meta = result["metadata"]
            citations += (
                f"- {meta['chapter']} | "
                f"{meta['heading']} "
                f"(Pages {meta['start_page']}-{meta['end_page']})\n"
            )

        assistant_message["content"] += citations
        yield history


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🩺 AIRA - Anesthesia Resident Assistant")

    chatbot = gr.Chatbot(height=500)

    with gr.Row():
        msg = gr.Textbox(placeholder="Ask about anesthesia...", scale=8)
        send_btn = gr.Button("Send", scale=1)

    with gr.Row():
        stop_btn = gr.Button("Stop")
        clear_btn = gr.Button("Clear")

    send_event = msg.submit(generate_response, [msg, chatbot], chatbot)
    send_btn_event = send_btn.click(generate_response, [msg, chatbot], chatbot)

    stop_btn.click(fn=None, cancels=[send_event, send_btn_event])

    clear_btn.click(lambda: [], None, chatbot)

demo.queue()
demo.launch(share=True)