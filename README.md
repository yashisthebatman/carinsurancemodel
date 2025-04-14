# 🚗 Car Insurance Claim Prediction System

An AI-powered web application that predicts the likelihood of a car insurance claim based on policy and user-specific parameters. Built to help users make informed decisions and assist insurance providers in identifying high-risk policies and reducing fraud.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)

---

## 🧠 Overview

In the modern insurance industry, fast and accurate claim risk analysis is essential. This system leverages Machine Learning to assess a customer's likelihood of claiming car insurance, enabling both transparency for users and cost-efficiency for insurers.

---

## ✨ Features

- 🔐 **User Authentication** – Secure login and registration
- 📥 **Data Input** – Policy ID, vehicle damage status, annual premium, and more
- 🧠 **ML Prediction** – Backend model trained on real-world insurance data
- 📊 **Graphical Insights** – Visual display of key predictive features
- 💾 **Database Storage** – Logs all user input and prediction history
- 🔁 **Real-time Feedback** – Instant likelihood result upon form submission

---

## 💻 Tech Stack

### 🔹 Frontend
- HTML, CSS, JavaScript
- Bootstrap (for responsive design)

### 🔹 Backend
- Python (Flask)
- REST APIs
- SQLite / PostgreSQL (for database)

### 🔹 Machine Learning
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn (for visualizations)
- Pickle / Joblib (for model deployment)

---

## 🧱 Architecture

```mermaid
graph TD
A[User Input via Web Form] --> B[Backend API - Flask]
B --> C[ML Model (model.pkl)]
B --> D[Database (SQLite/PostgreSQL)]
C --> E[Prediction + Confidence Score]
D --> F[Data Logging + Retrieval]
E --> G[Frontend Result Page with Graphs]
