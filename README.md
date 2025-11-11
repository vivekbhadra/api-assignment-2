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

```markdown
# Major Fixes and Enhancements Implemented

## 1. Persistent Streamlit Session and Tab State
**Problem:** Each time a form was submitted or a file was uploaded, Streamlit re-executed the entire script, automatically switching the interface back to the first tab (“Draft New Agreement”). This caused user workflow interruptions and loss of context during review.  

**Fix:** Replaced `st.tabs()` with a session-controlled `st.radio` selector and maintained the selected tab through `st.session_state.active_tab`. A lightweight helper function `set_tab()` was added to update the state programmatically during interactions such as file uploads or form submissions.  

**Impact:** The interface now remains on the selected tab (“Review & Fix Agreement” or “Draft New Agreement”) even after re-runs, ensuring a smooth and uninterrupted user experience.

---

## 2. Scoped Form Submission Handling
**Problem:** The variable `submitted` was declared outside its logical scope, causing runtime errors when the app switched to the review tab (where the variable was undefined).  

**Fix:** Moved the `if submitted:` logic fully inside the “Draft New Agreement” section to keep it context-aware and isolate the two workflows.  

**Impact:** Prevented `NameError` exceptions, improved code clarity, and aligned the control flow with Streamlit’s reactive execution model.

---

## 3. Streamlit Session Persistence for Review Results
**Problem:** Uploaded documents and generated review outputs were lost whenever the user switched tabs or re-uploaded files.  

**Fix:** Introduced `st.session_state` variables to persist uploaded files and intermediate results between app re-runs.  

**Impact:** The review analysis (summary, risk classification, and amendment suggestions) now persists across navigation, eliminating redundant reprocessing.

---

## 4. Correct Clause Formatting in Word Output
**Problem:** The generated Word document rendered entire paragraphs in bold instead of only the clause titles, leading to poor readability.  

**Fix:** Enhanced the regex parsing logic in `create_formatted_agreement()` to bold only clause headings such as “WHEREAS”, “NOW THEREFORE”, or numbered sections while keeping the body text normal.  

**Impact:** Legal agreements now adhere to standard formatting conventions and improved readability.

---

## 5. Structured LLM Metrics and Logging
**Problem:** There was no unified mechanism to monitor API usage, latency, or costs for model calls.  

**Fix:** Implemented a comprehensive `log_metric()` function that records latency, token usage, and cost into a structured JSONL file (`metrics_log.jsonl`). Added corresponding counters and summaries in the Streamlit sidebar for real-time visibility.  

**Impact:** Enabled consistent tracking of Gemini and OpenAI API performance, improved observability, and supported transparent cost monitoring.

---

## 6. Voice-to-Clause Integration
**Fix:** Integrated OpenAI Whisper for audio transcription and GPT-4o-mini for automatic conversion of spoken statements into legal clauses.  

**Impact:** Adds accessibility and productivity, allowing users to draft clauses verbally and receive polished legal text automatically.

---

## 7. Model Caching and Resource Optimisation
**Fix:** Applied the `@st.cache_resource` decorator to the DistilBERT risk-classification model to prevent reloading it on every app refresh.  

**Impact:** Significantly reduced startup time and improved overall app responsiveness.

---

## 8. Improved Error Handling and User Transparency
- Enhanced validation for mandatory input fields (landlord, tenant, and property details).  
- Added detailed error messages for API failures and malformed documents.  
- Displayed latency and cost metrics after each generation step to improve user trust and system transparency.

---

## 9. Backward Compatibility and Non-Disruptive Refactor
**Fix:** The fixes were implemented without altering any existing functionality or breaking backward compatibility.  
Core modules for document generation, logging, and review were preserved exactly as before; only tab handling and scope corrections were introduced.  

**Impact:** Ensures continuity in evaluation and consistent behaviour across both local and deployed versions.

---

## 10. Future-Ready Modular Architecture
**Fix:** Code structure has been logically separated into reusable functions (`create_formatted_agreement`, `log_metric`, `set_tab`, etc.), ready for future modularisation into independent Python modules such as `metrics.py`, `formatting.py`, and `review.py`.  

**Impact:** This design supports eventual migration to a microservice or container-based architecture under Docker/Kubernetes.
```


**Team Note:**  
All members are expected to be **available and responsive** over the next few days to ensure smooth completion and coordination of the project.

