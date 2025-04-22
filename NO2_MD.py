import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os

class LoanDefaultTrainer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.model = None
        self.pipeline = None

    def load_data(self):
        df = pd.read_csv(self.csv_path)
        df.dropna(inplace=True)
        X = df.drop(columns=["loan_status"])
        y = df["loan_status"]
        return X, y

    def build_pipeline(self):
        categorical_features = ["person_gender", "person_education", "person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"]
        numeric_features = ["person_age", "person_income", "person_emp_exp", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score"]
        
        transformer = ColumnTransformer(transformers=[
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='passthrough')

        self.pipeline = Pipeline(steps=[
            ("preprocessor", transformer),
            ("classifier", XGBClassifier(random_state=42))
        ])

    def train(self):
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.build_pipeline()
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {acc:.4f}")
        self.model = self.pipeline

    def save_model(self, output_path="loan_model.pkl"):
        with open(output_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {output_path}")

if __name__ == "__main__":
    trainer = LoanDefaultTrainer(csv_path="Dataset_A_loan.csv")
    trainer.train()
    trainer.save_model()
