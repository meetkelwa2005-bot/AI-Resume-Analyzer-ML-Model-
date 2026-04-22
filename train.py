import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
def create_mock_data():
    """Generates mock resume data for training."""
    data = {
        "text": [
            "Experienced software engineer with 5 years in Python, Java, and scalable backend services.",
            "Frontend developer proficient in React, HTML, CSS, and modern UI/UX design.",
            "Data scientist with expertise in machine learning, Python, Pandas, and deep learning models.",
            "DevOps engineer skilled in AWS, Docker, Kubernetes, and CI/CD pipelines.",
            "Backend engineer focusing on Node.js, Express, MongoDB, and API development.",
            "Full stack developer with experience in Vue.js, Django, PostgreSQL, and AWS.",
            "Machine learning engineer who builds predictive models using TensorFlow and PyTorch.",
            "Cloud architect experienced with AWS infrastructure, Terraform, and cloud security.",
            "Software developer specialized in C++, system performance, and algorithmic optimization.",
            "UI/UX designer with a strong portfolio in Figma, Adobe XD, and user research."
        ],
        "category": [
            "Backend Engineering",
            "Frontend Engineering",
            "Data Science",
            "DevOps",
            "Backend Engineering",
            "Full Stack Engineering",
            "Data Science",
            "DevOps",
            "Backend Engineering",
            "Design"
        ]
    }
    return pd.DataFrame(data)
def main():
    print("Initializing training pipeline...")
    
    # 1. Load data
    df = create_mock_data()
    print(f"Loaded {len(df)} mock records.")
    
    # 2. Preprocessing & Vectorization
    X = df['text']
    y = df['category']
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)
    
    # Split data (using a small test size just for demonstration)
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
    
    # 3. Model Definition and Training
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    print("\nTraining complete. Model Evaluation:")
    # Using zero_division=0 to prevent warnings on small mock dataset
    print(classification_report(y_test, predictions, zero_division=0))
    
    # 4. Save the trained weights and vectorizer
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "model.pkl")
    vectorizer_path = os.path.join("models", "vectorizer.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved successfully to {model_path}")
    print(f"Vectorizer saved successfully to {vectorizer_path}")
if __name__ == "__main__":
    main()
