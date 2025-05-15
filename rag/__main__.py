if __name__ == "__main__":
    from . import Main, SOCKET_HOST, SOCKET_PORT, log_info

    m = Main()

    import socketio
    from aiohttp import web

    # Create a Socket.IO server
    sio = socketio.AsyncServer(cors_allowed_origins="*")  # Allow any origin (CORS)
    app = web.Application()
    sio.attach(app)

    # Handle a new connection
    @sio.event
    async def connect(sid, environ):
        print(f"Client connected: {sid}")

    # Handle messages
    @sio.event
    async def message(sid, data):
        print(f"Message from {sid}: {data}")
        chat_id = data.get("chat_id")
        chat_summary = data.get("chat_summary")
        customer_message = data.get("customer_message")
        customer_emotion = data.get("customer_emotion")

        async def on_new_token(token: str | None):
            #             is_finished: Type.Boolean(),
            # error: Type.Boolean(),
            # data: Type.String(),
            if token is not None:
                data = {
                    "is_finished": False,
                    "error": False,
                    "data": token,
                }
            else:
                data = {
                    "is_finished": True,
                    "error": False,
                    "data": "",
                }
            await sio.emit("message-response", data, room=sid)

        await m.rag.answer(
            chat_id, chat_summary, customer_message, customer_emotion, on_new_token
        )

    @sio.event
    async def ter(sid, data):
        print(f"TER from {sid}: {data}")
        text = data.get("text")
        ser_emotion = data.get("ser_emotion")

        if ser_emotion:
            prompt = f"""
                Bạn là một chuyên gia phân tích cảm xúc văn bản. Bạn còn rất am hiểu cách hoạt động của hệ thống xác định cảm xúc qua giọng nói (Speech-Emotion Recognition - SER).
                Biết rằng hệ thống SER được huấn luyện trên tập dữ liệu khá mất cân bằng, dẫn đến việc nó có thể phân loại sai cảm xúc của người nói ; nó thường xuyên dự đoán cảm xúc tiêu cực thay vì tích cực.

                Vậy, hãy xác định cảm xúc của khách hàng dựa vào cảm xúc dự đoán của hệ thống SER cũng như dựa vào của văn bản dưới đây.
                Bạn sẽ phân loại chúng theo 5 nhãn: "rất tích cực", "tích cực", "trung tính", "tiêu cực", "rất tiêu cực".
                Định dạng câu trả lời của bạn: chỉ gồm 1 trong 5 nhãn nêu trên, không bao gồm các dấu nháy (").
                
                Cảm xúc của khách hàng theo hệ thống SER là: {ser_emotion}
                Văn bản như sau: {text}
                """
        else:
            prompt = f"""    Hãy xác định cảm xúc của văn bản dưới đây theo 5 nhãn: "rất tích cực", "tích cực", "trung tính", "tiêu cực", "rất tiêu cực".
                Định dạng câu trả lời của bạn: chỉ gồm 1 trong 5 nhãn nêu trên, không bao gồm các dấu nháy (").
                
                Văn bản như sau: {text}"""
        result = m.rag.llm_for_ter.call(prompt)
        log_info(f"TER result: {result}")
        await sio.emit("ter-response", result, room=sid)

    # Handle disconnection
    @sio.event
    async def disconnect(sid):
        print(f"Client disconnected: {sid}")

    # Run the web app
    web.run_app(app, host=SOCKET_HOST, port=SOCKET_PORT)
