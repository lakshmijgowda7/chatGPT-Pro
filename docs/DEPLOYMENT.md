# 24/7 Cloud Deployment Guide (Laptop Off / Shutdown Safe)

When running on your laptop with `localhost` or `localtunnel`, the app goes offline as soon as your laptop is closed, shut down, or disconnected from Wi-Fi.

To make this app **accessible 24/7 by your friends on mobile and laptops even when your laptop is completely powered off**, follow one of the 100% free cloud deployment methods below.

---

## 🌟 Method 1: Render.com (Recommended — 100% Free & Easiest)

Render hosts both your FastAPI backend and Next.js frontend in the cloud for free with automatic SSL (`https://`). We have pre-configured [`render.yaml`](file:///c:/Users/laksh/OneDrive/Desktop/LLM%20XRAY/render.yaml) in this repository so deployment is 1-click!

### Step 1: Push your project to GitHub
1. Create a free account on [GitHub.com](https://github.com/) (if you don't have one).
2. Create a new GitHub repository named `chatgpt-pro`.
3. Push your project code to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Deploy ChatGPT Pro"
   git branch -M main
   git remote add origin https://github.com/<your-username>/chatgpt-pro.git
   git push -u origin main
   ```

### Step 2: Deploy on Render.com
1. Go to [Render.com](https://render.com/) and sign up for free (or Sign In with GitHub).
2. Click **New +** > **Blueprint**.
3. Select your `chatgpt-pro` GitHub repository.
4. Render will automatically detect `render.yaml` and show:
   - **`chatgpt-pro-backend`** (FastAPI)
   - **`chatgpt-pro-frontend`** (Next.js)
5. Click **Apply**.
6. Render will automatically build and launch both services!
7. Once finished, Render gives you your permanent live URL:
   `https://chatgpt-pro-frontend.onrender.com`

> [!TIP]
> This link will stay online **24 hours a day, 7 days a week**, completely independent of your laptop! Anyone can open it on their phone, iPad, or computer at any time.

---

## 🚀 Method 2: Vercel (Frontend) + Render (Backend)

Vercel is the fastest host for Next.js web applications, offering instant global delivery and zero latency.

### Step 1: Deploy Backend to Render or Railway
1. In [Render.com](https://render.com), click **New +** > **Web Service**.
2. Connect your GitHub repository and set:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables**:
      - `LLM_PROVIDER`: `groq`
      - `LLM_API_KEY`: `your-groq-api-key-here`
      - `LLM_MODEL`: `qwen/qwen3.8-27b`
      - `CORS_ORIGINS`: `["*"]`
3. Copy your backend's live URL (e.g. `https://my-backend.onrender.com`).

### Step 2: Deploy Frontend to Vercel
1. Go to [Vercel.com](https://vercel.com/) and sign in with GitHub.
2. Click **Add New...** > **Project** and import your repository.
3. Set **Root Directory** to `frontend`.
4. Add Environment Variable:
   - `NEXT_PUBLIC_API_URL`: `https://my-backend.onrender.com/api/v1`
5. Click **Deploy**.
6. In ~60 seconds, Vercel gives you your permanent, lightning-fast public link:
   `https://chatgpt-pro.vercel.app`

---

## 🔥 Method 3: Firebase Hosting + Google Cloud Run

To use Google's official cloud infrastructure:
1. Double-click [`deploy_firebase.bat`](file:///c:/Users/laksh/OneDrive/Desktop/LLM%20XRAY/deploy_firebase.bat) in the project directory.
2. The frontend will be hosted on Google's global CDN at:
   `https://<your-firebase-project>.web.app`

---

## Summary of Cloud vs Localhost

| Feature | Localhost / Tunnel | Cloud Deployment (Render / Vercel) |
|---|---|---|
| **Laptop Closed / Shutdown** | ❌ Goes Offline | ✅ **Stays Online 24/7** |
| **Accessible by Friends** | ⚠️ Only when your laptop is awake | ✅ **Always accessible worldwide** |
| **Mobile Phone Access** | ⚠️ Requires laptop running | ✅ **Works on any phone 24/7** |
| **URL Stability** | ⚠️ Changes when restarted | ✅ **Permanent fixed URL** |
| **Cost** | Free | **100% Free** |
