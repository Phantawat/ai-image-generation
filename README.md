# AI Entertainment - LoRA Image Generator

This project is a part of AI-Enabled subject coursework.
A modern, "Cyberpunk" aesthetic web application for generating AI images using Stable Diffusion with LoRA adaptors. This project features a React frontend utilizing Tailwind CSS v4 for styling and a Flask backend for handling image generation requests.

![application screenshot](/backend/static/generated/image.png)


## ✨ Features

- **Text-to-Image Generation**: Generate high-quality images from text prompts.
- **LoRA Support**: Designed to work with Low-Rank Adaptation models for stylized outputs.
- **Modern UI/UX**: 
  - Dark mode "Cyberpunk/Zinc" aesthetic.
  - Responsive Grid Layout.
  - Interactive "iPhone-style" Gallery.
  - Terminal-style History log.
- **Real-time Status**: Polling mechanism with visual progress indicators.
- **Gallery & History**: Client-side simulated gallery and history management (Mock data implementation).

## 🛠️ Tech Stack

### Frontend
- **Framework**: React 19 (via Vite)
- **Styling**: Tailwind CSS v4
- **Icons**: Lucide React
- **HTTP Client**: Axios
- **Language**: TypeScript

### Backend
- **Server**: Flask
- **ML Framework**: PyTorch, Diffusers
- **Model**: Stable Diffusion (via Hugging Face Diffusers)

---

## 🚀 Getting Started

### Prerequisites
- **Node.js** (v18+)
- **Python** (v3.10+)
- **NVIDIA GPU** (Recommended for local inference, though CPU fallback is possible but slow)

---

### 1. Backend Setup

Navigate to the backend directory and set up the Python environment.

```bash
cd backend

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** Ensure you have the necessary LoRA model files in `backend/lora-output/` if required by `app.py`, or ensure the script autodownloads the base model.

### 2. Frontend Setup

Navigate to the frontend directory and install Node dependencies.

```bash
cd frontend

# Install dependencies
npm install

# Ensure Tailwind v4 plugin is set up
npm install -D @tailwindcss/vite
```

---

## 🏃‍♂️ Running the Application

You need to run both the backend server and the frontend development server simultaneously.

### Terminal 1: Start Backend (Flask API)

```bash
cd backend
# With venv activated:
python app.py
```
*The backend usually runs on `http://localhost:5000`*

### Terminal 2: Start Frontend (Vite)

```bash
cd frontend
npm run dev
```
*The frontend usually runs on `http://localhost:5173`*

Open your browser and navigate to the link provided by Vite (e.g., `http://localhost:5173`).

---

## 📂 Project Structure

```
ai-image-project/
├── backend/               # Flask API & ML Model Logic
│   ├── app.py             # Main application entry point
│   ├── predict.py         # Inference logic
│   ├── train.py           # LoRA training script
│   ├── requirements.txt   # Python dependencies
│   ├── data/              # Metadata & training data
│   ├── lora-output/       # Trained adapters
│   └── static/generated/  # Output directory for images
│
└── frontend/              # React Application
    ├── src/
    │   ├── App.tsx        # Main UI Logic (Gallery, History, Create)
    │   ├── main.tsx       # Entry point
    │   └── index.css      # Tailwind imports
    ├── package.json
    └── vite.config.ts
```

## 📝 Usage Guide

1. **Create Tab**: Enter a prompt (e.g., "A cyberpunk city in rain") and click **Generate Artwork**.
2. **Progress**: Watch the circular progress indicator as the image generates.
3. **Gallery Tab**: View previously generated images. Hover (or tap) to reveal Download and Copy Prompt buttons.
4. **History Tab**: View a log of your past prompts. Click "Reuse" to load a prompt back into the generator.

## 🤝 Contributing

Feel free to fork this project and submit Pull Requests. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

[MIT](https://choosealicense.com/licenses/mit/)
