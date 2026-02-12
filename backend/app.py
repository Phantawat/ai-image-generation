import os
import uuid
import queue
import threading
import torch
import json
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

# ----------------------------
# Configuration
# ----------------------------
UPLOAD_FOLDER = 'static/generated'
DATA_FOLDER = 'data'
JOBS_FILE = os.path.join(DATA_FOLDER, 'jobs.json')
LORA_PATH = "./lora_output"  # Point this to your trained folder
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# Initialize JSON DB if not exists
if not os.path.exists(JOBS_FILE):
    with open(JOBS_FILE, 'w') as f:
        json.dump([], f)

app = Flask(__name__)
CORS(app)

# ----------------------------
# Global State & Locks
# ----------------------------
job_queue = queue.Queue()
jobs = {}  # In-memory fast cache for polling
pipeline = None
db_lock = threading.Lock() # Prevents file corruption

# ----------------------------
# Database Helpers
# ----------------------------
def read_db():
    """Reads the JSON database safely."""
    with db_lock:
        try:
            with open(JOBS_FILE, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

def write_db(data):
    """Writes to the JSON database safely."""
    with db_lock:
        with open(JOBS_FILE, 'w') as f:
            json.dump(data, f, indent=2)

def add_job_to_db(job_data):
    """Adds a new job to the beginning of the list."""
    current_data = read_db()
    current_data.insert(0, job_data)
    write_db(current_data)

def update_job_in_db(job_id, updates):
    """Updates a specific job in the JSON file."""
    current_data = read_db()
    found = False
    for job in current_data:
        if job['id'] == job_id:
            job.update(updates)
            found = True
            break
    if found:
        write_db(current_data)

# ----------------------------
# 1. Model Loader
# ----------------------------
def load_pipeline():
    global pipeline
    print("⏳ Loading Model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipeline = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        safety_checker=None
    )
    
    if os.path.exists(LORA_PATH):
        print(f"✨ Loading LoRA from {LORA_PATH}")
        try:
            pipeline.load_lora_weights(LORA_PATH)
        except Exception as e:
            print(f"⚠️ Failed to load LoRA: {e}")

    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.to(device)
    print("✅ Model Ready.")

# ----------------------------
# 2. The Worker Thread
# ----------------------------
def worker():
    print("👷 Worker thread started")
    while True:
        job_id = job_queue.get()
        try:
            process_job(job_id)
        except Exception as e:
            print(f"❌ Job {job_id} failed: {e}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            update_job_in_db(job_id, {"status": "failed", "error": str(e)})
        finally:
            job_queue.task_done()

def process_job(job_id):
    prompt = jobs[job_id]["prompt"]
    neg_prompt = jobs[job_id].get("negative_prompt", "")
    
    print(f"🎨 Processing: {prompt}")
    
    # Update Status
    jobs[job_id]["status"] = "processing"
    update_job_in_db(job_id, {"status": "processing"})

    # Run Generation
    image = pipeline(
        prompt=prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=25,
        guidance_scale=7.5
    ).images[0]

    # Save to Disk
    filename = f"{job_id}.png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image.save(filepath)

    # Update Data
    result_url = f"/static/generated/{filename}"
    
    # Update In-Memory
    jobs[job_id]["status"] = "completed"
    jobs[job_id]["result_url"] = result_url

    # Update DB
    update_job_in_db(job_id, {
        "status": "completed",
        "image_filename": filename,
        "completed_at": datetime.now().isoformat()
    })
    
    print(f"✅ Finished: {job_id}")

# ----------------------------
# 3. API Endpoints
# ----------------------------

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt")
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    job_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # 1. Add to In-Memory (Fast Access)
    jobs[job_id] = {
        "status": "queued",
        "prompt": prompt,
        "negative_prompt": data.get("negative_prompt", ""),
    }
    
    # 2. Add to JSON Database (Persistence)
    db_record = {
        "id": job_id,
        "status": "queued",
        "prompt": prompt,
        "negative_prompt": data.get("negative_prompt", ""),
        "created_at": timestamp,
        "image_filename": None
    }
    add_job_to_db(db_record)

    job_queue.put(job_id)
    
    return jsonify({"job_id": job_id, "status": "queued"})

@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    # Check in-memory first for speed
    job = jobs.get(job_id)
    
    # If not in memory (e.g. server restarted), check DB
    if not job:
        all_jobs = read_db()
        job_record = next((item for item in all_jobs if item["id"] == job_id), None)
        if job_record:
             # Reconstruct simple response
             job = {
                 "status": job_record["status"], 
                 "result_url": f"/static/generated/{job_record['image_filename']}" if job_record.get('image_filename') else None
             }

    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    response = {"status": job["status"], "id": job_id}
    
    if job["status"] == "completed":
        response["image_url"] = request.host_url.rstrip('/') + job["result_url"]
    elif job["status"] == "failed":
        response["error"] = job.get("error")

    return jsonify(response)

@app.route("/history", methods=["GET"])
def get_history():
    """Returns all jobs, newest first."""
    return jsonify(read_db())

@app.route("/gallery", methods=["GET"])
def get_gallery():
    """Returns only completed jobs with images."""
    all_jobs = read_db()
    gallery = [job for job in all_jobs if job.get("status") == "completed" and job.get("image_filename")]
    return jsonify(gallery)

# Start background worker
threading.Thread(target=worker, daemon=True).start()

# Load model
load_pipeline()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)