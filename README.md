🛡️ Inappropriate Message Detection (ML + Streamlit)

Live Demo:
👉 https://inappropriatemessagedetection-j8nh3mvzrrbbqt72fhpzuh.streamlit.app/

A machine-learning powered web application that detects cyberbullying, abusive, or harmful text using Natural Language Processing.
The system uses three ML models — Logistic Regression, Decision Tree, and Random Forest — and displays individual predictions, confidence scores, and a final aggregated decision.
The frontend features a modern, custom UI with a gradient banner, navbar, message box, and alert components.

📌 Project Highlights

🔍 Detects cyberbullying in real time

🧠 Three-model prediction system (Logistic Regression, Decision Tree, Random Forest)

📊 Confidence scores for each model

🟢 Beautiful custom UI (gradient design, rounded cards, result alerts)

💾 Models serialized with Joblib

⚙️ End-to-end ML pipeline with training script

☁️ Fully deployed on Streamlit Cloud

📂 Clean project architecture




🖼️ UI Preview
✨ 1. Input Page (User enters text)

The user enters any sentence to check whether it contains cyberbullying



<img width="1895" height="782" alt="image" src="https://github.com/user-attachments/assets/b74bcc0a-0423-4a68-8910-85e0aea8fc3d" />



🚨 2. Output Page (Prediction Results)

Displays:
✔ Final bullying / not bullying result
✔ Confidence level
✔ Predictions from all ML models
✔ Aggregated confidence (mean)



<img width="1913" height="811" alt="image" src="https://github.com/user-attachments/assets/04bb2095-c02e-4252-9423-807562de7dd6" />




🧠 How It Works
1️⃣ Preprocessing

Clean the text

Convert to lowercase

Tokenize and vectorize using TF-IDF

2️⃣ Models Used

The app uses three ML models trained on the dataset:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Each model outputs:

Prediction: Bullying / Not Bullying

Confidence Score (%)

3️⃣ Aggregated Result

The final decision is based on:

Combined confidence (mean of all models)

Majority voting logic (if applied)

Displayed visually with a colored alert box



    🛢️ Dataset

   messages.csv


🌐 Deployment (Streamlit Cloud)

The system is deployed on Streamlit Cloud.

Steps:

Push repo to GitHub

Open https://share.streamlit.io

Choose repository

Set main file path → app.py

Deploy

Live demo link (working):
👉 https://inappropriatemessagedetection-j8nh3mvzrrbbqt72fhpzuh.streamlit.app/

🛠️ Tech Stack

Python

Streamlit (UI + deployment)

Scikit-learn

TF-IDF Vectorizer

Logistic Regression

Decision Tree

Random Forest

Pandas / NumPy

🎯 Why This Project Is Strong for Recruiters

Demonstrates complete ML workflow

Includes multiple models + comparison

Strong UI/UX sense with custom design

Fully deployed, easy to test

Shows ability to integrate ML with frontend

Real-world problem: cyberbullying detection

🖥️ User Interaction Flow

User enters a message into the input box

The text is vectorized using the TF-IDF model

All three ML models generate:

Prediction (bullying / not bullying)

Confidence score

Confidence scores are averaged

A final decision is displayed on the screen:

Cyberbullying Detected! (red alert)

No Cyberbullying Detected. (green alert)

🤝 Contributing

Contributions are welcome!
Open an issue or submit a pull request.



📚 Future Enhancements

This project can be extended in multiple directions:

🔄 Add deep learning NLP models (BERT, DistilBERT, LSTM)

🌐 Integrate with social media APIs (Twitter, Discord, YouTube)

🧑‍💻 Build an admin dashboard for monitoring flagged messages

📊 Add visual analytics for bullying trends

🗣️ Add multilingual support

🤖 Deploy as an API microservice (FastAPI / Flask)

🧵 Include sentiment analysis for better context understanding



📄 License

This project is open-source under the MIT License.

