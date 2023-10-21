import argparse
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
        # Get image data and text prompts from the JSON data in the POST request
        data = request.json
        image_data = data.get("image_data")
        config = data.get("config")
        grounded_checkpoint = data.get("grounded_checkpoint")
        sam_checkpoint = data.get("sam_checkpoint")
        input_image = data.get("input_image")
        output_dir = data.get("output_dir")
        box_threshold = float(data.get("box_threshold", 0.3))
        text_threshold = float(data.get("text_threshold", 0.25))
        det_prompt = data.get("det_prompt")
        inpaint_prompt = data.get("inpaint_prompt")
        device = data.get("device", "cpu")

        # Convert base64 image data to an image
        image = Image.open(BytesIO(base64.b64decode(image_data))

        # Save the image to a temporary file
        temp_image_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp_image_file, format="JPEG")
        image_path = temp_image_file.name

        # Execute the Python script with the provided image and text prompts
        subprocess.call([
            "python", "grounded_sam_inpainting_demo.py",  # Replace with the actual path to your Python script
            "--config", config,
            "--grounded_checkpoint", grounded_checkpoint,
            "--sam_checkpoint", sam_checkpoint,
            "--input_image", image_path,
            "--output_dir", output_dir,
            "--box_threshold", str(box_threshold),
            "--text_threshold", str(text_threshold),
            "--det_prompt", det_prompt,
            "--inpaint_prompt", inpaint_prompt,
            "--device", device,
            # Add more arguments as needed
        ])

        # Read the output image
        output_image_path = os.path.join(output_dir, "grounded_sam_inpainting_output.jpg")
        output_image = Image.open(output_image_path)

        # Convert the output image to base64
        output_image_data = base64.b64encode(output_image.tobytes()).decode('utf-8')

        return response.json({'output_image': output_image_data})

    except Exception as e:
        return response.json({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
