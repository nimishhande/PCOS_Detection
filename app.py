from flask import Flask, render_template, request
import pandas as pd, joblib, xgboost as xgb, os

app = Flask(__name__)

# ───────────────────────────────
# 1)  Load model + preprocessor
# ───────────────────────────────
pre, _, _ = joblib.load(os.path.join("model", "preprocessed.joblib"))
model      = joblib.load(os.path.join("model", "pcos_xgb.model"))

FEATURES = [
    "Age","BMI","Fast_food","LH","FSH","PRL","AMH",
    "Vit_D3","BP_Systolic","BP_Diastolic",
    "Blood_group","Cycle_length","Pregnant"
]

# ───────────────────────────────
# 2)  Routes
# ───────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/checker", methods=["GET", "POST"])
def checker():
    prediction = None
    if request.method == "POST":
        # Collect form values
        data = {f: request.form.get(f) for f in FEATURES}
        # Convert numeric fields
        for key in data:
            if key not in ["Blood_group"]:
                data[key] = float(data[key])
        df = pd.DataFrame([data])[FEATURES]
        X  = pre.transform(df)
        prob = float(model.predict(xgb.DMatrix(X))[0])
        if prob < 0.40:
                 prediction = f"Low Risk ({prob*100:.1f}%) – Keep up the good lifestyle!"
        elif 0.40 <= prob <= 0.65:
                prediction = f"Moderate Risk ({prob*100:.1f}%) – Take care of your habits and monitor symptoms. Discipline your health."
        else:
                 prediction = f"High Risk ({prob*100:.1f}%) – Consult a doctor and undergo a full diagnosis."

    return render_template("checker.html", prediction=prediction)

@app.route("/info")
def info():
    return render_template("info.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/disclaimer")
def disclaimer():
    return render_template("disclaimer.html")


app.run(host="0.0.0.0", port=8080)
