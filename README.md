# SmartLegal Rental Assistant – AWS EC2 Deployment (Complete Setup)

### Version: November 2025  
**Author:** Vivek Bhadra  
**Repository:** [SmartLegal Rental Assistant](https://github.com/vivekbhadra/api-assignment-2)  
**Domain:** LegalTech (Rental Agreement Automation)  

---

## Overview

The **SmartLegal Rental Assistant** is an AI-powered web application built with Streamlit.  
It helps automate rental agreement drafting and legal review under India’s **Model Tenancy Act 2021**.

This guide covers a **complete, reproducible setup** on AWS EC2 — from instance creation to accessing the app in your browser.  
All steps are tested on **Ubuntu 22.04 LTS (t4g.large / t3.medium)** in the **eu-west-2 (London)** region.

---

## Features

1. **Draft New Rental Agreement** – Generated using Gemini 2.0 Flash (Google AI).  
2. **Review & Suggest Amendments** – Legal validation powered by Gemini + BERT.  
3. **Summarise Uploaded Agreement** – Extract key details.  
4. **Voice to Clause** – Uses OpenAI Whisper + GPT-4o-mini for speech-to-text.  
5. **Formatted Word Document Output** – Generated rental agreement saved in `.docx`.

---

## Step 1: Launch AWS EC2 Instance

### Open AWS Console → EC2 → Launch Instance
- **Name:** `SmartLegalRentalApp`
- **AMI:** `Ubuntu Server 22.04 LTS (64-bit, x86 or ARM)`
- **Instance Type:** `t4g.large` (recommended) or `t3.medium`
- **Storage:** 30 GiB gp3
- **Key Pair:** Select or create one (e.g. `SmartLegalKey.pem`)
- **Security Group Rules:**
  | Type | Port | Source | Purpose |
  |------|------|--------|----------|
  | SSH | 22 | My IP | Connect via terminal |
  | HTTP | 80 | Anywhere (0.0.0.0/0) | Web access |
  | HTTPS | 443 | Anywhere (0.0.0.0/0) | Secure web access |
  | Custom TCP | 8501 | Anywhere (0.0.0.0/0) | Streamlit app port |

---

## Step 2: Add User Data Script

Scroll to **Advanced Details → User Data** and paste the following script:

```bash
#!/bin/bash
# === SmartLegal Rental Assistant Setup ===

# Update system and install dependencies
apt update -y
apt install -y python3 python3-pip git ffmpeg

# Create project directory
mkdir -p /home/ubuntu/api-assignment-2
cd /home/ubuntu/api-assignment-2

# Clone the GitHub repo
git clone https://github.com/vivekbhadra/api-assignment-2.git .
chown -R ubuntu:ubuntu /home/ubuntu/api-assignment-2

# Create virtual environment
pip install virtualenv
python3 -m venv rentalenv
source rentalenv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install streamlit google-generativeai openai python-dotenv PyPDF2 python-docx torch transformers pillow

# Create READY flag
echo "SmartLegal Rental Assistant setup complete!" > /home/ubuntu/READY.txt
```
### This Script Performs the Following

- Installs **Python**, **pip**, **git**, and **FFmpeg**.  
- Clones the **GitHub repository**.  
- Creates and activates a **Python virtual environment**.  
- Installs all necessary **dependencies automatically**.  

When setup completes, a file `/home/ubuntu/READY.txt` will be created.

---

## Step 3: Connect to EC2 Instance

Once the EC2 instance is running, connect to it from your local terminal:

```bash
cd ~/Downloads
chmod 400 SmartLegalKey.pem
ssh -i SmartLegalKey.pem ubuntu@<EC2_PUBLIC_IP>
```
Example
```
ssh -i SmartLegalKey.pem ubuntu@18.133.78.52
```
Check Readiness
```
cat /home/ubuntu/READY.txt
```

Expected Output
```
SmartLegal Rental Assistant setup complete!
```
## Step 4: Set Up Environment Variables

Inside your EC2 instance, navigate to the app directory and create a .env file:  
```
cd ~/api-assignment-2
nano .env
```

Create your API keys from here:   
https://aistudio.google.com/app/api-keys    
https://platform.openai.com/api-keys    

Add your API keys:
```
GEMINI_API_KEY=<your_gemini_api_key>
OPENAI_API_KEY=<your_openai_api_key>
```

Save and exit (Ctrl + O, Enter, then Ctrl + X).  

## Step 5: Activate Virtual Environment

Activate the Python virtual environment:  
```
cd ~/api-assignment-2
source rentalenv/bin/activate
```

Confirm the correct environment is active:  
```
which python
```

Expected Output  
```
/home/ubuntu/api-assignment-2/rentalenv/bin/python
```
## Step 6: Run the Application

Start the Streamlit app:
```
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

Expected Output  

You can now view your Streamlit app in your browser.
```
URL: http://0.0.0.0:8501

Network URL: http://<EC2_PUBLIC_IP>:8501
```
## Step 7: Access the App in Browser

On your local computer, open a web browser and visit:
```
http://<EC2_PUBLIC_IP>:8501
```
Example
```
http://18.133.78.52:8501
```

You should now see the **SmartLegal Rental Assistant** interface with:

- **Tab 1:** Draft New Agreement  
- **Tab 2:** Review & Fix Agreement  
- **Sidebar:** Voice to Clause  

---

# Step 8: Stop or Restart the App

To stop the running Streamlit process:

Press `Ctrl + C` in the terminal.

To restart the app:

```
source rentalenv/bin/activate
cd ~/api-assignment-2
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```
### Developer Notes
Region: eu-west-2 (London)  
Recommended Instance Type: t4g.large (ARM)  
Default Port: 8501  
App Directory: /home/ubuntu/api-assignment-2  

View Logs:    
```
tail -f ~/.streamlit/logs/*  
```

# Work Distribution List

**Project Coordination Message (10 Nov 2025)**

## 1. Vivek Bhadra
- Overall coordination and deployment.  
- Currently handling **deployment setup and related tasks**.

## 2. Mayank
- Responsible for **project video recording**.  
- Prepare recording **setup and environment**.  
- Test the recording on personal setup to ensure readiness.
- Once the project is completed, you will be in charge of metrics collection and recording with screenshots and log traces as applicable.

## 3. Swagat
- In charge of **end-to-end application testing**.  
- Verify full functionality, workflows, and integration.
- Create Test Cases, plan and document the test cases with screenshots and logs where applicable.
- Check CloudWatch Logs and try capturing CloudWatch logs as evidence of the sequence of things happening during the execution.
- Create Flow diagram from the CloudWatch Logs

## 4. Sidharth
- Responsible for **application aesthetics and UI polish**:
  - Adjust **font size, font type, and formatting** for consistency.  
  - Fix **bug in the “Review Agreement” section**.  
  - Ensure visual and user experience improvements.

---

# SmartLegal App — End-to-End Deployment on AWS EKS
This section lists every successful step, from Docker image build to full Kubernetes deployment, including the .env secret handling and the final deployment reapply and restart sequence.

## Build and push Docker image to Amazon ECR
```
# Authenticate Docker with your ECR registry
aws ecr get-login-password --region eu-west-2 | docker login --username AWS --password-stdin 402691950139.dkr.ecr.eu-west-2.amazonaws.com

# Build the image
docker build -t smartlegal-app .

# Tag the image for ECR
docker tag smartlegal-app:latest 402691950139.dkr.ecr.eu-west-2.amazonaws.com/smartlegal-app:latest

# Push it to ECR
docker push 402691950139.dkr.ecr.eu-west-2.amazonaws.com/smartlegal-app:latest
```

## Create and verify EKS cluster
```
eksctl create cluster \
  --name smartlegal-cluster-v2 \
  --region eu-west-2 \
  --version 1.32 \
  --nodegroup-name smartlegal-nodes-v2 \
  --nodes 2 \
  --nodes-min 2 \
  --nodes-max 3 \
  --node-type t3.large \
  --managed
```

Verify:  
```
aws eks list-clusters --region eu-west-2
aws eks update-kubeconfig --region eu-west-2 --name smartlegal-cluster-v2
kubectl get nodes
```

## Create Kubernetes Secret from .env
Ensure the .env file is in the project root and contains your API keys:  
```
OPENAI_API_KEY=<your-openai-key>
GEMINI_API_KEY=<your-gemini-key>
```
Create the secret:  
```
kubectl create secret generic smartlegal-env --from-env-file=.env
```
Verify:  
```
kubectl get secret smartlegal-env
```
## Prepare Deployment YAML
```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: smartlegal-deployment
  namespace: default
spec:
  replicas: 1                   # temporarily single pod for file downloads
  selector:
    matchLabels:
      app: smartlegal
  template:
    metadata:
      labels:
        app: smartlegal
    spec:
      containers:
      - name: smartlegal
        image: 402691950139.dkr.ecr.eu-west-2.amazonaws.com/smartlegal-app:latest
        ports:
        - containerPort: 8501
        env:
        - name: PORT
          value: "8501"
        - name: GOOGLE_API_KEY            # map Gemini key to expected variable
          valueFrom:
            secretKeyRef:
              name: smartlegal-env
              key: GEMINI_API_KEY
        envFrom:
        - secretRef:
            name: smartlegal-env
```

## Prepare Service YAML
```
apiVersion: v1
kind: Service
metadata:
  name: smartlegal-service
spec:
  type: LoadBalancer
  selector:
    app: smartlegal
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8501
```

## Deploy to EKS
```
kubectl apply -f smartlegal-deployment.yaml
kubectl apply -f smartlegal-service.yaml
```
Monitor rollout:  
```
kubectl get pods -w
kubectl get svc smartlegal-service
```
## Verify environment variables inside pod
```
kubectl exec -it <pod-name> -- env | grep OPENAI
kubectl exec -it <pod-name> -- env | grep GOOGLE
```
## Reapply and Restart Deployment (after Secret fix)
After correcting the environment mapping (GOOGLE_API_KEY → GEMINI_API_KEY), the deployment was re-applied and restarted.  
```
# Reapply updated deployment file
kubectl apply -f smartlegal-deployment.yaml

# Restart the deployment to ensure pods load new environment variables
kubectl rollout restart deployment smartlegal-deployment

# Verify rollout
kubectl get pods
```
## Verify 
```
kubectl get svc smartlegal-service
```
## Access the app
Open the LoadBalancer URL in browser:  
```
http://a177958bd700d4965926d8ca3c883da8-1598551454.eu-west-2.elb.amazonaws.com
```

## Fix for Document Download Issue
When multiple replicas were running, users received an error like:  

“The generated document could not be downloaded.”  
Streamlit served requests through the AWS LoadBalancer, which distributed them across multiple pods.  
Each pod had its own isolated filesystem, so the .docx file generated on one pod was not accessible to another pod when the download request was routed there. 

```
vim smartlegal-deployment.yaml
```
Change:  
```
spec:
  replicas: 2
```
to:
```
spec:
  replicas: 1
```
Apply the update and restart the deployment:
```
kubectl apply -f smartlegal-deployment.yaml
kubectl rollout restart deployment smartlegal-deployment
```
Verify:
```
kubectl get pods
```
All user requests now go to the same pod, ensuring the generated .docx files remain available during download.  
Document download works reliably from the Streamlit interface.

## Store files in S3
For a production or scalable solution, it’s better to upload generated files to S3 instead of serving them directly.
That way:

All pods can upload to the same bucket.  
The user always downloads from S3 (public link).  
You can scale to multiple replicas safely.  
Here’s the change you’d make inside your Streamlit app:  
```
import boto3
import streamlit as st
from datetime import datetime

# Configure S3 client (IAM role on EKS node already allows access)
s3 = boto3.client("s3")
BUCKET_NAME = "smartlegal-generated-files"

def upload_to_s3(file_path):
    filename = file_path.split("/")[-1]
    s3.upload_file(file_path, BUCKET_NAME, filename)
    return f"https://{BUCKET_NAME}.s3.eu-west-2.amazonaws.com/{filename}"

file_path = f"Formatted_Rental_{tenant_name}.docx"
s3_url = upload_to_s3(file_path)
st.success("File successfully generated and uploaded.")
st.markdown(f"[Download {file_path}]({s3_url})", unsafe_allow_html=True)
```
Then create the S3 bucket once:
Now, every generated .docx is uploaded and can be downloaded reliably from any pod.

## Load Time Optimisation
The SmartLegal app initially took several minutes to load when deployed on EKS.  
This delay was caused by a large Docker image (~4.5 GB), slow container startup, Streamlit’s library initialisation overhead, and LoadBalancer warm-up time.

### Analysis
* The key contributors to the slow startup were:  
* The original image included GPU-enabled torch and transformers packages, which added gigabytes of unnecessary CUDA and cuDNN binaries.  
* Each container had to pull the massive image from ECR before starting.  
* Streamlit loaded all libraries (including AI models) at launch instead of lazily on demand.  
* The AWS LoadBalancer required several minutes to initialise and register healthy targets.  

### Optimisation Steps
* Slimmed the Docker Image:  
* Replaced python:3.10 base image with python:3.10-slim.  
* Removed unused tools and dependencies (git, curl, etc.).  
* Created a clean requirements.txt limited to necessary libraries only.  
* Reintroduced Torch and Transformers (CPU-only):  
* Installed lightweight CPU wheels from the PyTorch CPU repository:

```
torch==2.3.0+cpu
transformers==4.44.0
--find-links https://download.pytorch.org/whl/cpu/torch_stable.html
```
This kept AI functionality intact while reducing image size from ~4.5 GB to ~1.2 GB.

### Optimised Docker Build Process  
* Used layer caching by copying and installing requirements.txt first.  
* Removed the apt-get upgrade step to avoid unnecessary system bloat.  
* Used --no-cache-dir in pip installs to prevent wheel caching inside the image.

### Improved Streamlit Initialisation  
Added caching for heavy imports using:
```
@st.cache_resource
def load_clients():
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    openai.api_key = os.getenv("OPENAI_API_KEY")
    return genai, openai
```
This ensures large models and clients are only initialised once per container.

### Deployment Improvements
* Used a single replica for consistent performance during testing.  
* Retained the LoadBalancer but accepted its initial warm-up delay (only affects first rollout).

After these optimisations, the application starts and becomes accessible within seconds of rollout.  
Subsequent restarts or updates now require minimal downtime, and the EKS cluster no longer experiences heavy image-pulling delays.  

## LLMOps Metrics and Observability 
To ensure the SmartLegal Rental Assistant adheres to LLMOps principles, several performance and operational metrics were integrated into the system. These metrics allow continuous monitoring of model responsiveness, reliability, and resource efficiency. The design objective was to measure at least five relevant metrics — namely latency, token count, cost estimation, model confidence, and successful query count — in line with the assignment requirement.  

### Latency Measurement
Latency represents the total time taken by a generative AI model to process an input prompt and return a response. It serves as a key performance indicator of model responsiveness.  
The script records timestamps before and after each model call using the time module, then computes the elapsed time.  
```
start_time = time.time()
response = model.generate_content(prompt)
end_time = time.time()
latency = round(end_time - start_time, 2)
```
This metric is calculated for every major AI operation — including agreement drafting, summary generation, and amendment suggestion. It is then displayed to the user through Streamlit’s caption interface:  
```
st.caption(f"Latency: {latency}s | Tokens: {token_count} | Cost: £{cost}")
```
The latency value helps evaluate the model’s efficiency and responsiveness under various prompt sizes.  

### Token Count Estimation
The token count approximates the total number of tokens (input + output) processed by the model. While approximate, it provides a useful proxy for computational workload and helps estimate cost.  
```
token_count = len(prompt.split()) + len(draft.split())
```
This counts the number of whitespace-separated words in both the prompt and the generated output. Although this is a rough estimate, it closely reflects the number of processed tokens in most English-language LLMs.
Token count is particularly relevant when calculating the overall cost per query, as API billing is typically token-based.  

### Cost Estimation
Cost estimation provides awareness of the approximate monetary cost incurred per API call. It reinforces responsible AI use by allowing users to see how model interaction scales with input size.  
```
cost = round(token_count * 0.0005 / 1000, 6)
```
A constant multiplier (0.0005) is used to represent a hypothetical cost-per-token factor in British pounds. Dividing by 1000 gives an estimated cost for the transaction.  
Displaying this value in the Streamlit interface encourages transparency and awareness of operational expenditure.   
Example output:  
```
Latency: 4.23s | Tokens: 1482 | Cost: £0.00074
```
### Model Confidence (Risk Classification Score)
For the “Review & Fix Agreement” module, a BERT-based classifier was integrated to assess the legal risk level of the uploaded agreement.  
The confidence score from this model represents the classification confidence — an essential metric for evaluating accuracy and reliability.  
```
risk = risk_pipe(text[:10000])[0]
st.write(f"**{risk['label']}** (Confidence: {risk['score']:.1%})")
```
The pipeline returns both a label (“SAFE” or “RISKY”) and a confidence score between 0 and 1.  
This quantitative value provides a direct indicator of model certainty, fulfilling the requirement for a quality or accuracy-related metric.  
Example display:  
```
Risk Level: RISKY (Confidence: 82.5%)
```
### Successful Query Count
This metric tracks how many AI queries have been successfully executed during the current user session. It reflects application reliability and user interaction volume.  
```
if "success_count" not in st.session_state:
    st.session_state.success_count = 0
st.session_state.success_count += 1
st.sidebar.metric("Total Successful Queries", st.session_state.success_count)
```
By using st.session_state, the value persists throughout the Streamlit session. Each successful AI call increments the counter, and the cumulative total is displayed on the sidebar as a dynamic metric:  
```
Total Successful Queries: 7
```
This provides a lightweight observability mechanism for tracking real-time usage during demonstrations or testing.  

### Metric Display and Observability
All collected metrics are integrated into the Streamlit interface, ensuring immediate visibility. The use of captions and sidebar metrics enhances interpretability without needing external dashboards.  
For instance, after a successful generation task, the following line renders the real-time metrics below the output text area:  
```
st.caption(f"Latency: {latency}s | Tokens: {token_count} | Cost: £{cost}")
```
Similarly, risk classification confidence and total query count appear within separate submodules of the application.  

## LLMOps Significance
The integration of these five metrics transforms the application from a simple AI utility into an observable and measurable LLMOps system.
It enables:
* Real-time tracking of model performance.
* Transparency in cost and efficiency.
* Quantitative assessment of model accuracy and application reliability.
This approach aligns with modern LLMOps practices, where each AI interaction is monitored not only for functional correctness but also for operational health and cost-effectiveness.
Sample Output (as Displayed in the Application):

<img width="240" height="724" alt="image" src="https://github.com/user-attachments/assets/b73f9adf-1fe8-4e47-a78a-cc4eccf9ec79" />  

<img width="1832" height="78" alt="image" src="https://github.com/user-attachments/assets/f610e13d-28bc-496a-932c-1ffc84661256" />


# Project–Assignment Mapping

## Domain Selected
The domain chosen for this project is **LegalTech**, focusing on the automation of **residential rental agreements in India** under the *Model Tenancy Act, 2021*.  
This theme is reflected throughout the application — from the prompts used to draft the agreement to the document formatting, tab names, and legal references that appear in the user interface.

---

## AI Categories Used
The project combines two categories of Artificial Intelligence: **Natural Language Processing (NLP)** and **Speech Recognition**.

- The **NLP** component is responsible for generating the rental agreement, providing a 100-word summary, suggesting amendments, and performing risk classification.  
  This is handled through Google’s *Gemini 2.0 Flash* model and a *DistilBERT* classifier.

- The **Speech Recognition** part allows users to speak a clause and have it automatically transcribed and converted into formal legal text.  
  This is achieved using *OpenAI Whisper* for transcription and *GPT-4o-mini* for converting the transcribed text into a properly worded clause.

---

## Core Functional Sub-Tasks
The project is structured around five coherent and interdependent sub-tasks, all aimed at simplifying the drafting and review of rental agreements:

1. **Drafting** a complete legal agreement using the Gemini model.  
2. **Reviewing** the text and suggesting amendments as per the Model Tenancy Act.  
3. **Summarising** the agreement in concise, readable form.  
4. **Classifying** the agreement as SAFE or RISKY based on its clauses using the DistilBERT model.  
5. **Voice-to-Clause conversion**, allowing a spoken statement to be added as a legal clause.

Each of these sub-tasks is implemented within the same Streamlit app:
- Drafting is done in the *“Draft New Agreement”* tab.  
- Review, summary, and amendment suggestions are handled in the *“Review & Fix Agreement”* tab.  
- Risk classification is performed by the `load_risk_model()` function.  
- The voice-to-clause feature is integrated in the sidebar.

---

## Models and APIs Integrated
The application makes use of multiple AI services and models working together:
- **Gemini 2.0 Flash** is used for all generative tasks such as drafting, summarisation, and amendment suggestions.  
- **DistilBERT (SST-2)** provides baseline risk classification.  
- **OpenAI Whisper** performs speech-to-text transcription.  
- **GPT-4o-mini** reformats the spoken text into a legal clause.

All models are configured and loaded in `app.py`, with API keys securely managed through environment variables and `.env` configuration.

---

## Unified Objective
All components serve one overarching goal: to assist landlords and tenants in producing a legally compliant, readable, and professionally formatted rental agreement.  
The workflow begins with generating a draft, moves through automated review and risk assessment, and ends with an exportable `.docx` document created by the `create_formatted_agreement()` function.

---

## API-Based, Deployable Application
The solution is designed as an **interactive Streamlit web application** that communicates with multiple AI APIs.  
It has been fully **containerised using Docker** and **deployed on AWS EKS (Elastic Kubernetes Service)**.  
The build and deployment process is automated via the `build_and_deploy.sh` script, which handles Docker image creation, ECR push, and Kubernetes rollout verification.

Deployment validation is done using `kubectl` commands, confirming that pods and the public LoadBalancer service are active.

---

## LLMOps and Observability
A key feature of this project is its emphasis on **LLMOps metrics** for transparency and operational insight.  
For every AI call, the system records:
- Latency (in seconds)  
- Token usage  
- Estimated cost (in GBP)  
- Request status (success or failure)  
- Average latency and failure rate over time  

These metrics are logged to `metrics_log.jsonl` and also displayed live in the Streamlit sidebar.  
The `log_metric()` function standardises this process for all core operations, including drafting, summarisation, amendment review, risk classification, and voice-to-clause generation.

---

# Fine-tuning the Legal Risk Classifier (CPU)

Create and activate a virtual environment:  
```
cd ~/api-assignment-2
python3 -m venv .venv
source .venv/bin/activate
```
Upgrade pip (recommended):  
```
python -m pip install --upgrade pip
```
Install required libraries:  
```
pip install torch transformers pandas accelerate
```
Verify the environment: 
```
python check_training_env.py
```
Create the training dataset:  
```
mkdir -p data
cat > data/rental_risk_dataset.csv <<'CSV'
text,label
"The landlord may evict the tenant without notice if rent is delayed by 5 days.",RISKY
"The landlord and tenant agree to a one-month notice period for termination.",SAFE
"Security deposit will be forfeited entirely upon minor damage.",RISKY
"The landlord must refund the security deposit within fifteen days after lease termination.",SAFE
"Tenant is responsible for all maintenance including structural repairs.",RISKY
"The landlord shall handle major repairs while the tenant covers minor maintenance.",SAFE
"The tenant must pay property tax.",RISKY
"Either party may terminate the tenancy by giving one-month written notice.",SAFE
"Rent shall be increased by 20% every six months without negotiation.",RISKY
"The rent shall be revised only by mutual agreement once a year.",SAFE
```
Run fine-tuning and capture logs: 
```
python fine_tune_rental_risk.py \
  --csv data/rental_risk_dataset.csv \
  --outdir models/rental-risk-bert \
  --epochs 3 \
  --batch 8 \
  --lr 5e-5 \
  --maxlen 256 \
  | tee fine_tune_log.txt
```
Confirm artefacts: 
```
ls -1 models/rental-risk-bert
# Expect: config.json, pytorch_model.bin, tokenizer.json, train_metrics.json (and related files)

cat models/rental-risk-bert/train_metrics.json
```

---

## Summary
This project presents a cohesive, AI-driven workflow — combining natural language generation, legal reasoning, speech processing, and monitoring — within a single, deployable web application.  
Every sub-task contributes directly to the central objective of creating, reviewing, and refining legally compliant rental agreements in an efficient and user-friendly manner.


**Team Note:**  
All members are expected to be **available and responsive** over the next few days to ensure smooth completion and coordination of the project.

