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

## Step 1 — Launch AWS EC2 Instance

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

## Step 2 — Add User Data Script

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

## Step 3 — Connect to EC2 Instance

Once the EC2 instance is running, connect to it from your local terminal:

```bash
cd ~/Downloads
chmod 400 SmartLegalKey.pem
ssh -i SmartLegalKey.pem ubuntu@<EC2_PUBLIC_IP> 
