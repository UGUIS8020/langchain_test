import gradio as gr
from dotenv import load_dotenv
from chatbot_engine import chat, create_index
from langchain.memory import ChatMessageHistory


def respond(message, chat_history):
    history = ChatMessageHistory()
    for [user_messege, ai_message] in chat_history:
        history.add_user_message(user_messege)
        history.add_ai_message(ai_message)

    bot_message = chat(message, history, index)
    chat_history.append((message, bot_message))
    return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])    

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    load_dotenv()

    index = create_index()

    demo.launch()
    
# test

    # 認証が必要なとき
# if __name__ == "__main__":
# load_dotenv()

#     app_env = os.environ.get("APP_ENV", "production")

#     if app_env == "production":
#         username = os.environ["GRADIO_USERNAME"]
#         password = os.environ["GRADIO_PASSWORD"]
#         auth = (username, password)
#     else:
#         auth = None

#     demo.launch(auth=auth)