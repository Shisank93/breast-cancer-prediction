# OncoPredict AI: Production Deployment Guide

This guide details three industry-standard paths to deploy the **OncoPredict AI** web application to the internet. 

Because we have pre-packaged a production-ready `Dockerfile` and structured the application cleanly, it can be deployed with minimal configuration.

---

## Option 1: Render (Simplest & Free Web Service)
Render is a cloud platform that allows hosting web applications directly from a GitHub repository. It supports both Python and Docker runtimes.

### Steps:
1. **Push your code to GitHub**:
   - Create a repository (e.g. `breast-cancer-prediction`) and push your local directory.
2. **Sign up for Render**:
   - Go to [render.com](https://render.com/) and sign up (linking your GitHub account).
3. **Create a New Web Service**:
   - Click **New +** on the dashboard and select **Web Service**.
   - Connect your GitHub repository.
4. **Configure Settings**:
   - **Name**: `oncopredict-ai` (or any preferred name)
   - **Environment**: Select **Docker** (Render will automatically detect the `Dockerfile` at the root, build the image, run the training pipeline inside it, and expose the server!).
   - **Region**: Select the region closest to your users.
   - **Instance Type**: Select the **Free** tier.
5. **Deploy**:
   - Click **Create Web Service**. 
   - Render will build the Docker container and deploy it. Once the build log shows `INFO: Application startup complete.`, your app is live!

---

## Option 2: Hugging Face Spaces (Best for ML Portfolios)
Hugging Face Spaces provides free hosting for machine learning demos. It natively supports custom Docker containers and is a great addition to a data science resume.

### Steps:
1. **Create a Hugging Face Account**:
   - Sign up at [huggingface.co](https://huggingface.co/).
2. **Create a New Space**:
   - Go to Spaces, click **Create new Space**.
   - **Space Name**: `breast-cancer-prediction`
   - **SDK**: Select **Docker**.
   - **Template**: Choose **Blank** (or any simple starter).
   - **Visibility**: Public (free CPU basic tier).
3. **Upload/Commit Files**:
   - Hugging Face spaces are Git repositories. Clone your space repository locally, copy your project files into it, and push.
   - Alternatively, upload the files directly via the Hugging Face web interface:
     - `Dockerfile`, `app/`, `src/`, `config/`, `static/`, `templates/`, `requirements.txt`, `setup.py`.
4. **How the Space runs**:
   - Hugging Face will read the `Dockerfile`, install dependencies, run `verify_pipeline_phase3.py` to train the model, and launch the FastAPI app.
   - **Important**: Hugging Face Spaces expect web servers to run on port `7860`.
   - To make it work, modify the last line of the `Dockerfile` from `--port 8000` to `--port 7860` before pushing to Hugging Face, or set an environment variable `PORT=7860` in the Space settings.

---

## Option 3: Google Cloud Run (Enterprise Standard)
Google Cloud Run is a managed serverless platform that automatically scales containerized applications. It is free for low traffic and highly robust.

### Prerequisites:
- Install [Google Cloud CLI](https://cloud.google.com/sdk/gcloud) locally.
- A Google Cloud Platform (GCP) project with billing enabled.

### Steps:
1. **Initialize gcloud**:
   ```bash
   gcloud init
   gcloud auth login
   ```
2. **Enable Required APIs**:
   ```bash
   gcloud services enable artifactregistry.googleapis.com run.googleapis.com
   ```
3. **Build and Submit Container to Google Artifact Registry**:
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/oncopredict-ai:latest
   ```
4. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy oncopredict-ai \
     --image gcr.io/YOUR_PROJECT_ID/oncopredict-ai:latest \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```
5. **Get your live URL**:
   - Once completed, the CLI will output a live URL (e.g. `https://oncopredict-ai-xxxxx.a.run.app`).

---

## Post-Deployment Checklist
- [ ] Test the `/health` endpoint to verify the model and scaler are loaded properly.
- [ ] Fill out the form and submit a prediction to check database logging and results pages.
- [ ] Inspect dashboard statistics under `/metrics`.
