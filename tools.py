import cv2
import base64
from dotenv import load_dotenv

load_dotenv()

def capture_image() -> str:
    """
    Capture one fram the default webcam, resizes it,
    encodes it as base64 JPEG (raw string) and return it.
    """
    for idx in range(10):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            for _ in range(10): # Warm up
                cap.read()
            ret, fram = cap.read()
            cap.release()
            if not ret:
                continue
            cv2.imwrite("sample.jpg", fram) #Optional
            ret, buf = cv2.imencode('.jpg', fram)
            if ret:
                return base64.b64encode(buf).decode('utf-8')
    raise RuntimeError("Could not open any webcame (tried indices 0-3)")

# capture_image()

from groq import Groq

def analyze_image_with_query(query: str) -> str:
    """
    Expecting a string with 'query'
    Capture the image ans sends the query and returns the analysis.
    """

    img_b64 = capture_image()
    # print(img_b64)
    model = "meta-llama/llama-4-maverick-17b-128e-instruct"

    if not query or not img_b64:
        return "Error: both 'query' and 'image' fields required"
    
    client = Groq()
    messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    },
                },
            ],
        }]
    
    chat_completion = client.chat.completions.create(
        messages = messages,
        model = model
    )
    return chat_completion.choices[0].message.content

# query = "How many people do you see "
# print(analyze_image_with_query(query))
