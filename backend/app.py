from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from model import VideoProcessor
from utils import Dehaze, ProcessVideo
import io
import cv2
import os
import uuid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'net', 'trained_models', 'its_train.pk')
IMAGE_DIR = os.path.join(BASE_DIR, 'image')
os.makedirs(IMAGE_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

video_processor = VideoProcessor(model_path=MODEL_PATH)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "service": "Image & Video Dehazing API"
    })

@app.route("/dehaze", methods=["POST"])
def dehaze():
    try:
        dehaze_type = request.form.get('type')
        model = request.form.get('model')

        if not dehaze_type or not model:
            return jsonify({"error": "Missing parameters. Both 'type' and 'model' are required."}), 400

        if dehaze_type not in ['image', 'video']:
            return jsonify({"error": "Invalid type. Must be 'image' or 'video'."}), 400

        if model not in ['dcp', 'ffa']:
            return jsonify({"error": "Invalid model. Must be 'dcp' or 'ffa'."}), 400

        req_id = uuid.uuid4().hex[:8]

        if dehaze_type == 'image':
            if 'file' not in request.files:
                return jsonify({"error": "No image file provided"}), 400

            image = request.files['file']
            if image.filename == '':
                return jsonify({"error": "No selected image file"}), 400

            ext = os.path.splitext(image.filename)[1] or '.jpg'
            image_path = os.path.join(IMAGE_DIR, f"input_{req_id}{ext}")
            output_path = os.path.join(IMAGE_DIR, f"dehazed_{req_id}.jpg")
            image.save(image_path)

            try:
                if model == 'dcp':
                    dehaze_image = Dehaze(image_path)
                    cv2.imwrite(output_path, dehaze_image * 255)
                elif model == 'ffa':
                    dehazed_image = video_processor.process_image(image_path)
                    cv2.imwrite(output_path, dehazed_image)

                with open(output_path, 'rb') as f:
                    output_bytes = f.read()

                return send_file(
                    io.BytesIO(output_bytes),
                    mimetype="image/jpeg",
                    as_attachment=False,
                    download_name="dehazed_image.jpg"
                )

            except Exception as e:
                return jsonify({"error": f"Error processing image with {model.upper()}: {str(e)}"}), 500
            finally:
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except Exception:
                        pass
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except Exception:
                        pass

        elif dehaze_type == 'video':
            if 'file' not in request.files:
                return jsonify({"error": "No video file provided"}), 400

            video = request.files['file']
            if video.filename == '':
                return jsonify({"error": "No selected video file"}), 400

            temp_input = os.path.join(BASE_DIR, f"temp_input_{req_id}.mp4")
            temp_output = os.path.join(BASE_DIR, f"temp_output_{req_id}.mp4")

            try:
                if model == 'dcp':
                    video.save(temp_input)
                    ProcessVideo(temp_input, temp_output)

                    if not os.path.exists(temp_output):
                        raise Exception("Output video file was not created")

                    with open(temp_output, 'rb') as f:
                        processed_video = f.read()

                elif model == 'ffa':
                    video_bytes = video.read()
                    processed_video = video_processor.process_video(video_bytes)

                return send_file(
                    io.BytesIO(processed_video),
                    mimetype="video/mp4",
                    as_attachment=True,
                    download_name="dehazed_video.mp4"
                )

            except Exception as e:
                return jsonify({"error": f"Error processing video with {model.upper()}: {str(e)}"}), 500

            finally:
                if os.path.exists(temp_input):
                    try:
                        os.remove(temp_input)
                    except Exception:
                        pass
                if os.path.exists(temp_output):
                    try:
                        os.remove(temp_output)
                    except Exception:
                        pass

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
