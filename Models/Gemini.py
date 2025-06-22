import google.generativeai as genai
from PIL import Image
import numpy as np

def Gemini_15_flash(video_frames, question, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    
    images = [Image.fromarray(frame.astype(np.uint8)) for frame in video_frames]

    try:
        response = model.generate_content(
            contents=[
                {"role": "user", "parts": [question] + images}              ],
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=10
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"❌ Error: {e}")
        return "error"