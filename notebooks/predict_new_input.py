import joblib
import pandas as pd
from preprocessors import NumericPreprocessor, CategoricalPreprocessor  

# =============================
# LOAD MODELS
# =============================
PREPROCESSOR_PATH = "../models/preprocessor.pkl"
ENSEMBLE_PATH     = "../models/ensemble_voting.pkl"


preprocessor = joblib.load(PREPROCESSOR_PATH)
ensemble     = joblib.load(ENSEMBLE_PATH)

# =============================
# RAW FEATURES (ORDER MATTERS)
# =============================
FEATURES = [
    "academic_rank",
    "tenure_status",
    "institution_type",
    "years_at_institution",
    "base_salary",
    "teaching_load",
    "research_funding",
    "department_size",
    "admin_support",
    "work_life_balance",
    "promotion_opportunities",
    "publications_last_3_years",
    "student_evaluation_avg"
]

# =============================
# INPUT HELPERS
# =============================

def ask_option(prompt, options):
    print(f"\n📌 {prompt}")
    for i, opt in enumerate(options, 1):
        print(f"   {i}. {opt}")

    while True:
        try:
            choice = int(input("👉 Select option number: "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
        except:
            pass
        print("❌ Invalid choice. Try again.")


def ask_number(prompt, min_val, max_val, float_allowed=False):
    print(f"\n📌 {prompt}  (Allowed Range: {min_val} → {max_val})")
    while True:
        try:
            val = float(input("👉 Enter value: ")) if float_allowed else int(input("👉 Enter value: "))
            if min_val <= val <= max_val:
                return val
        except:
            pass
        print("❌ Invalid number. Try again.")


# =============================
# BUILD INPUT DATAFRAME
# =============================
def get_user_input():
    data = {}

    # --- CATEGORICAL ---
    data["academic_rank"] = ask_option(
        "Select Academic Rank:",
        ["Assistant Professor", "Associate Professor", "Full Professor", "Lecturer"]
    )

    data["tenure_status"] = ask_option(
        "Select Tenure Status:",
        ["Tenure-Track", "Tenured", "Non-Tenure"]
    )

    data["institution_type"] = ask_option(
        "Select Institution Type:",
        ["Technical Institute", "Research University", "Liberal Arts College", "Community College"]
    )

    # --- NUMERIC ---
    data["years_at_institution"] = ask_number("Years at Institution", 0, 30)
    data["base_salary"] = ask_number("Base Salary", 3000, 450000, float_allowed=True)
    data["teaching_load"] = ask_number("Teaching Load", 1, 9)
    data["research_funding"] = ask_number("Research Funding", -150000, 400000, float_allowed=True)
    data["department_size"] = ask_number("Department Size", 5, 50)
    data["admin_support"] = ask_number("Admin Support (1–9)", 1, 9)
    data["work_life_balance"] = ask_number("Work-Life Balance (1–9)", 1, 9)
    data["promotion_opportunities"] = ask_number("Promotion Opportunities (1–9)", 1, 9)
    data["publications_last_3_years"] = ask_number("Publications (Last 3 Years)", 0, 20)
    data["student_evaluation_avg"] = ask_number("Student Evaluation Avg", 1.0, 5.0, float_allowed=True)

    return pd.DataFrame([data], columns=FEATURES)


# =============================
# PREDICT FUNCTION
# =============================
def predict(df_raw):

    # 1️⃣ Try predicting with RAW DF (ensemble contains pipelines)
    try:
        pred = ensemble.predict(df_raw)[0]
        prob = ensemble.predict_proba(df_raw)[0][1]

    # 2️⃣ If that fails → preprocess manually then predict
    except Exception:
        print("\n⚠ Raw prediction failed — applying preprocessor...")

        X = preprocessor.transform(df_raw)

        if hasattr(X, "toarray"):
            X = X.toarray()

        pred = ensemble.predict(X)[0]
        prob = ensemble.predict_proba(X)[0][1]

    # Convert probability → percentage
    prob_percent = round(prob * 100, 2)

    status = "LEAVE (1)" if int(pred) == 1 else "STAY (0)"

    print("\n========================================")
    print("          🔮 FINAL PREDICTION")
    print("========================================")
    print(f"📌 Prediction: {status}")
    print(f"📊 Chance of Leaving: {prob_percent}%")
    print("========================================\n")


# =============================
# RUN THE CLI
# =============================
if __name__ == "__main__":
    
    df_input = get_user_input()
    predict(df_input)
