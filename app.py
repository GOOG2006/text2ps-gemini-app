import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from io import BytesIO
from werkzeug.utils import secure_filename

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure the Gemini API key
# Make sure to have GOOGLE_API_KEY in your .env file
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Fatal Error: Could not configure Gemini API. {e}")

# Configuration
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('templates', 'index.html')

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files (CSS, JS)."""
    return send_from_directory('static', path)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    instruction = request.form.get('instruction', '')
    if not instruction:
        return jsonify({'error': 'Instruction is required'}), 400

    if file and allowed_file(file.filename):
        # Create a unique filename to prevent overwrites
        original_filename = secure_filename(file.filename)
        unique_id = uuid.uuid4().hex
        filename = f"{unique_id}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            app.logger.info(f"Opening image for processing: {filepath}")
            img = Image.open(filepath)
            
            # Use the correct model that supports image editing
            app.logger.info("Initializing Generative Model (gemini-2.5-flash-image-preview)...")
            model = genai.GenerativeModel('gemini-2.5-flash-image-preview') 
            
            # The prompt can be simpler as the model is specialized
            prompt = instruction
            
            app.logger.info("Calling model.generate_content() with a 300s timeout... This may take a moment.")
            response = model.generate_content(
                [prompt, img],
                request_options={"timeout": 300}
            )
            app.logger.info("...model.generate_content() finished.")
            app.logger.info(f"Full Gemini Response: {response}")

            # --- Corrected Response Handling ---
            # According to the provided snippet, we need to parse the response differently.
            processed_image_data = None
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    processed_image_data = part.inline_data.data
                    break # Found the image, no need to look further
            
            if not processed_image_data:
                # Check for text part which might contain an error or refusal from the model
                text_response = ""
                for part in response.candidates[0].content.parts:
                    if part.text is not None:
                        text_response += part.text
                
                app.logger.error(f"Model did not return image. Text response: {text_response}")
                raise ValueError(f"The model did not return an image. Response: {text_response or 'Empty'}")

            app.logger.info("Extracting image data from response.")
            processed_image = Image.open(BytesIO(processed_image_data))
            
            # Save the processed image with a unique name
            # Ensure it saves in a web-compatible format like PNG
            processed_filename = f"processed_{filename.rsplit('.', 1)[0]}.png"
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            
            app.logger.info(f"Saving processed image to {processed_filepath}")
            processed_image.save(processed_filepath, 'PNG')
            
            processed_image_url = f'/uploads/{processed_filename}'

        except Exception as e:
            app.logger.error(f"An error occurred with the Gemini API: {e}")
            return jsonify({'error': f'Failed to process image with AI model. Details: {str(e)}'}), 500

        return jsonify({
            'original_url': f'/uploads/{filename}',
            'processed_url': processed_image_url,
            'instruction': instruction
        })

    return jsonify({'error': 'File type not allowed or an unexpected error occurred.'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)