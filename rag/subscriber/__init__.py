import redis
import json
from typing import Callable

def subscribe(host: str, port: int, message_handler: Callable[[str], None]):
    r = redis.Redis(host=host, port=port, decode_responses=True)
    pubsub = r.pubsub()

    def handle_message(message):
        if message['type'] == 'message':
            message_handler("")

    pubsub.subscribe(**{'my-channel': handle_message})

    print("Subscribed to 'my-channel'. Waiting for messages...")
    pubsub.run_in_thread(sleep_time=1.0)
    return r
