# Deployment Guide

## Option 1: Streamlit Community Cloud (Recommended)
The easiest way to deploy this app is using [Streamlit Community Cloud](https://streamlit.io/cloud).

1.  **Push to GitHub**: Ensure your code (including `requirements.txt`) is pushed to a GitHub repository.
2.  **Sign in**: Go to Streamlit Cloud and sign in with GitHub.
3.  **Deploy**:
    *   Click "New app".
    *   Select your repository, branch, and main file (`app.py`).
    *   Click "Deploy".
4.  **Secrets**:
    *   Go to your app's "Settings" -> "Secrets".
    *   Add your API keys in TOML format:
        ```toml
        OPENAI_API_KEY = "sk-..."
        GEMINI_API_KEY = "..."
        ```
    *   *Note: The app currently reads keys from the sidebar input or standard env vars. If deploying, you might want to modify the code to check `st.secrets` as well, or just input them in the UI every time.*

## Option 2: Docker
You can containerize the application for deployment on any platform that supports Docker (AWS ECS, Google Cloud Run, Azure Container Instances, etc.).

### Dockerfile
Create a `Dockerfile` in the root directory:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run
```bash
docker build -t agentic-semantic-mapper .
docker run -p 8501:8501 agentic-semantic-mapper
```

## Option 3: Manual Cloud Deployment (VM)
1.  Provision a VM (e.g., EC2, DigitalOcean Droplet).
2.  Clone the repo.
3.  Install Python and pip.
4.  Run `pip install -r requirements.txt`.
5.  Run `streamlit run app.py`.
    *   *Tip: Use `tmux` or `systemd` to keep it running.*
