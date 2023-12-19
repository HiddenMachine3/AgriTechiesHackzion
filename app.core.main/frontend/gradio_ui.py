import gradio as gr


def dosomething():
    return "hi"


# create ui interface to interact with gpt-3.5 model
iface = gr.Interface(
    fn=dosomething,
    inputs=gr.components.Textbox(lines=7, placeholder="Enter your question here"),
    outputs="text",
    title="Something",
    description="this is our input box",
)
iface.launch(share=False)
