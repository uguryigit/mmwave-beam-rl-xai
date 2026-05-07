import os
from datetime import datetime
import streamlit as st
import requests
import json
import time
import boto3
from botocore.exceptions import ClientError

st.set_page_config(page_title="Simulation Explain", layout="wide")
API_URL = os.environ.get("N8N_WEBHOOK_URL", "https://n8n.howlet.io/webhook/streamlit")

# ------- S3 settings -------
S3_BUCKET = os.environ.get("S3_BUCKET", "dqn-simulation")
S3_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_PREFIX = ""
S3_EXT = ""

# AWS credentials are read from the standard AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
# environment variables (or ~/.aws/credentials, or IAM role). See .env.example.
session = boto3.session.Session(region_name=S3_REGION)
s3 = session.client("s3")
 
with st.sidebar:
    st.sidebar.title("Upload Images")
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["png", "jpg", "jpeg", "pdf", "csv"],
        accept_multiple_files=True,
    )
    st.sidebar.title("Example Questions")
    st.write(
        "Which algorithm achieved the highest final evaluation reward, DQN or Q-factorized? Explain why using the simulation rules."
    )
 
st.title("Explaination of Simulation Results")
 
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
user_input = st.chat_input("Your question...")
 
def poll_s3_for_result(key: str, interval_sec: int = 5, max_attempts: int = 240):
    """
    S3'te {key} obje oluşana kadar interval_sec aralıklarla yoklar.
    Bulunca (text) içeriği döndürür. Bulamazsa None.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            body = obj["Body"].read()
            # İçerik text ise decode et; değilse raw döndür
            try:
                text = body.decode("utf-8")
                return text
            except UnicodeDecodeError:
                # Binary ise bir uyarı metni döndür (istersen burayı download linkine çevirebilirsin)
                return f"[Binary file of {len(body)} bytes received from S3 key: {key}]"
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404", "NotFound"):
                time.sleep(interval_sec)
                continue
            # Diğer hatalar: yetki vb.
            return f"S3 error while reading {key}: {e}"
    return None
 
if user_input:
    with st.chat_message("user"):
        st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
 
    with st.spinner("Generating response..."):
        # n8n'e isteği gönder
        file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        data = {"question": user_input, "output_file_name": file_name}
        files_payload = {}
        if uploaded_files:
            for i, uploaded_file in enumerate(uploaded_files):
                key = f"image-{i}"
                files_payload[key] = (uploaded_file.name, uploaded_file, uploaded_file.type)
 
        response = requests.post(API_URL, data=data, files=files_payload)
        response_json = response.json()
        # n8n tarafı anında bir text döndürüyorsa önce onu göster
        immediate = response_json.get("output")
 
    # Kullanıcıya anlık n8n çıktısı (varsa)
    if immediate:
        with st.chat_message("assistant"):
            st.write(immediate)
            st.session_state.messages.append({"role": "assistant", "content": immediate})
 
    # --- S3'ten 5 sn'de bir sonucu çek ---
    s3_key = f"{file_name}"
    with st.spinner(f"Waiting for S3 object: s3://{S3_BUCKET}/{s3_key} (polling every 5s)"):
        s3_result = poll_s3_for_result(s3_key, interval_sec=5, max_attempts=240) 
    if s3_result is None:
        msg = f"Timed out: Could not find s3://{S3_BUCKET}/{s3_key}."
        with st.chat_message("assistant"):
            st.write(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
    else:
        with st.chat_message("assistant"):
            st.write(s3_result)
            st.session_state.messages.append({"role": "assistant", "content": s3_result})
else:
    with st.chat_message("assistant"):
        st.write("How can I help you?")