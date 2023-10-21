import os
import io
from sanic import Sanic, response
from PIL import Image
from io import BytesIO
import base64
import subprocess
import tempfile

app = Sanic(__name)

@app.route('/inference', methods=['POST'])
async def inference(request):
    try:
        # Get image and text prompt from the request
        data = request.json
        image_data = data.get("image_data")
        text_prompt = data.get("text_prompt")

        # Convert base64 image data to an image
        image = Image.open(BytesIO(base64.b64decode(image_data))

        # Create the "outputs" folder if it doesn't exist
        if not os.path.exists("outputs"):
            os.makedirs("outputs")

        # Save the image to a temporary file
        temp_image_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp_image_file, format="JPEG")
        image_path = temp_image_file.name

        # Execute the Python script with the provided image and text prompt
        subprocess.call([
            "python", "grounded_sam_inpainting_demo.py",  # Replace with the actual path to your Python script
            "--input_image", image_path,
            "--det_prompt", text_prompt,
            # Add more arguments as needed
        ])

        # Read the output image
        output_image_path = os.path.join("outputs", "grounded_sam_inpainting_output.jpg")
        output_image = Image.open(output_image_path)

        # Convert the output image to base64
        output_image_data = base64.b64encode(output_image.tobytes()).decode('utf-8')

        return response.json({'output_image': output_image_data})

    except Exception as e:
        return response.json({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
