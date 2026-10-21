# Hugging Face Spaces: create a Docker Space, then upload the project and trained model.
FROM python:3.13-slim
WORKDIR /app
RUN useradd -m -u 1000 user
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user:user . /app
USER user
ENV HOME=/home/user
EXPOSE 7860
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true"]
