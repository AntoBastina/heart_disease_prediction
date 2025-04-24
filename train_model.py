import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load your dataset
df = pd.read_csv('heart_data.csv')  # Replace with your actual path

# Define features and label
X = df.drop('target', axis=1)
y = df['target']

# Define numerical and categorical features
numeric_features = ['age', 'resting_bp_s', 'cholesterol', 'max_heart_rate', 'oldpeak']
categorical_features = ['sex', 'chest_pain_type', 'fasting_blood_sugar',
                        'resting_ecg', 'exercise_angina', 'st_slope']

# Preprocessing setup
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define models
model_list = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "Decision_Tree": DecisionTreeClassifier(),
    "Random_Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

# Train and save each model
for name, model in model_list.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Split data (same split for consistency)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit pipeline
    pipeline.fit(X_train, y_train)
    
    # Save model
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    print(f"{name} model trained and saved as {name}.pkl")
