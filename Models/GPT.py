import openai
import base64
from io import BytesIO
from PIL import Image
import numpy as np

def encode_image(image: Image.Image) -> str:
 
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
def resize_video_frames(frames, size=(320, 240)):
    resized = []
    for frame in frames:
        img = Image.fromarray(frame)
        img = img.resize(size)
        resized.append(img)
    return resized

    
def GPT_4o_mini(video_frames_np: np.ndarray, question: str, api_key: str):
    import openai
    from PIL import Image
    from io import BytesIO
    import base64
    import numpy as np

    def encode_image(image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    client = openai.OpenAI(api_key=api_key)


    def resize_image(img: Image.Image, size=(360, 640)):
        return img.resize(size)

    images = [resize_image(Image.fromarray(f.astype(np.uint8))) for f in video_frames_np[:4]]

 
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                *[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encode_image(frame)}"
                        }
                    } for frame in images
                ]
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=10,
            temperature=0
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ GPT-4o-mini Error: {e}")
        return "error"