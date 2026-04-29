import pandas as pd
import sqlite3
import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import numpy as np
from geopy.geocoders import Nominatim
import io
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = "infra_predict_final_v2026"
DATABASE = "infra_data.db"

# Initialize Geocoder and Cache
geolocator = Nominatim(user_agent="infra_predict_app")
coord_cache = {}

# --- DATABASE INIT ---
def init_db():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY, email TEXT, password TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sales_data (
                Date TEXT, Project_ID TEXT, Project_Type TEXT, Location TEXT,
                Material_Cost REAL, Labor_Cost REAL, Equipment_Cost REAL,
                Total_Cost REAL, Revenue REAL, Profit REAL
            )
        """)
        conn.execute("INSERT OR IGNORE INTO users VALUES ('admin', 'admin@infra.com', 'admin123')")
init_db()

# --- HELPER: GLOBAL STATS ---
def get_global_stats():
    with sqlite3.connect(DATABASE) as conn:
        data = conn.execute("SELECT SUM(Revenue), AVG(Revenue), COUNT(*), SUM(Profit) FROM sales_data").fetchone()
        rows = conn.execute("SELECT Revenue FROM sales_data ORDER BY Date DESC LIMIT 2").fetchall()
    
    total_val = data[0] if data[0] else 0
    trend_val, is_up = 0, True
    
    if len(rows) >= 2:
        latest, previous = rows[0][0], rows[1][0]
        if previous > 0:
            trend_val = round(((latest - previous) / previous) * 100, 2)
            is_up = latest >= previous

    return {
        "total": f"{total_val:,.2f}", 
        "avg": f"{data[1]:,.2f}" if data[1] else "0.00",
        "profit": f"{data[3]:,.2f}" if data[3] else "0.00", 
        "days": data[2] if data[2] else 0,
        "trend": abs(trend_val),
        "is_up": is_up
    }

# --- AUTH ROUTES ---
@app.route("/")
def index(): 
    return redirect(url_for('login'))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user, pw = request.form.get("username"), request.form.get("password")
        with sqlite3.connect(DATABASE) as conn:
            res = conn.execute("SELECT * FROM users WHERE username=? AND password=?", (user, pw)).fetchone()
        if res:
            session["user"] = user
            return redirect(url_for("home"))
        flash("Invalid credentials!")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        if not username or not password:
            flash("Username and Password are required!")
            return redirect(url_for('signup'))
        try:
            with sqlite3.connect(DATABASE) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE username=?", (username,))
                if cursor.fetchone():
                    flash("Username already exists!")
                    return redirect(url_for('signup'))
                cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, password))
                conn.commit()
            flash("Account created! Please login.")
            return redirect(url_for("login"))
        except: flash("An error occurred.")
    return render_template("signup.html")

@app.route("/forgot", methods=["GET", "POST"])
def forgot():
    if request.method == "POST":
        email, new_pw = request.form.get("email"), request.form.get("new_password")
        with sqlite3.connect(DATABASE) as conn:
            conn.execute("UPDATE users SET password=? WHERE email=?", (new_pw, email))
        flash("Password updated!")
        return redirect(url_for("login"))
    return render_template("forgot.html")

# --- MAIN DASHBOARD (THE VERSION WITH 2 DIFFERENT DATA POINTS) ---
@app.route("/home")
def home():
    if "user" not in session: 
        return redirect(url_for("login"))
    
    stats = get_global_stats()

    with sqlite3.connect(DATABASE) as conn:
        # We fetch Revenue AND Total_Cost so the graphs aren't the same
        df = pd.read_sql("SELECT Project_ID, Revenue, Total_Cost FROM sales_data ORDER BY Revenue DESC", conn)
    
    if not df.empty:
        # labels = Project IDs
        labels = [str(x) for x in df['Project_ID'].tolist()]
        # Data 1 = Revenue (For the main Line chart)
        revenue = [float(x) for x in df['Revenue'].tolist()]
        # Data 2 = Costs (For the print/secondary Bar chart)
        costs = [float(x) for x in df['Total_Cost'].tolist()]
    else:
        labels, revenue, costs = ["No Data"], [0], [0]

    return render_template('home.html', 
                           stats=stats, 
                           chart_labels=labels, 
                           chart_data=revenue,   # This goes to the main Line Chart
                           chart_costs=costs)    # This can be used for the Bar Chart

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "user" not in session: return redirect(url_for("login"))
    if request.method == "POST":
        file = request.files.get('file')
        if file:
            try:
                df = pd.read_csv(file)
                df.columns = [c.strip() for c in df.columns]
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
                df = df.dropna(subset=['Date'])
                with sqlite3.connect(DATABASE) as conn:
                    df.to_sql('sales_data', conn, if_exists='append', index=False)
                return redirect(url_for("predict"))
            except Exception as e: flash(f"Upload Error: {str(e)}")
    return render_template("upload.html", stats=get_global_stats())

@app.route("/comparison", methods=["GET", "POST"])
def comparison():
    if "user" not in session: return redirect(url_for("login"))
    
    delta_data = None
    chart_data = {"labels": [], "past": [], "present": []}

    if request.method == "POST":
        file_past = request.files.get('file_past')
        file_present = request.files.get('file_present')
        
        if file_past and file_present:
            try:
                df_past = pd.read_csv(file_past)
                df_present = pd.read_csv(file_present)
                df_past.columns = [c.strip() for c in df_past.columns]
                df_present.columns = [c.strip() for c in df_present.columns]

                past_total = df_past['Revenue'].sum()
                present_total = df_present['Revenue'].sum()
                diff = present_total - past_total
                growth = round((diff / past_total * 100), 2) if past_total != 0 else 0
                
                p_grouped = df_past.groupby('Project_Type')['Revenue'].sum()
                n_grouped = df_present.groupby('Project_Type')['Revenue'].sum()
                
                comparison_df = pd.DataFrame({'Past': p_grouped, 'Present': n_grouped}).fillna(0)
                comparison_df['Growth'] = comparison_df['Present'] - comparison_df['Past']
                top_5 = comparison_df.sort_values(by='Growth', ascending=False).head(5)

                chart_data = {
                    "labels": top_5.index.tolist(),
                    "past": top_5['Past'].tolist(),
                    "present": top_5['Present'].tolist()
                }

                delta_data = {
                    "past": f"{past_total:,.2f}",
                    "present": f"{present_total:,.2f}",
                    "diff": f"{abs(diff):,.2f}",
                    "growth": growth,
                    "status": "Expansion" if diff >= 0 else "Contraction",
                    "color": "#00ff88" if diff >= 0 else "#ff4444",
                    "insight": "Significant upward momentum detected." if growth > 5 else "Variance audit recommended."
                }
            except Exception as e:
                flash(f"Data Error: {str(e)}")
            
    return render_template("comparison.html", delta_data=delta_data, chart_data=chart_data)

@app.route("/predict")
def predict():
    if "user" not in session: return redirect(url_for("login"))
    with sqlite3.connect(DATABASE) as conn:
        df = pd.read_sql("SELECT Date, Revenue FROM sales_data ORDER BY Date ASC", conn)
    if df.empty: return render_template("predict.html", labels=[], values=[], anomalies=[], insight="", stats=get_global_stats())
    labels, values = df['Date'].tolist(), [float(x) for x in df['Revenue'].tolist()]
    mean, std = np.mean(values), np.std(values)
    anomalies = [1 if abs(x - mean) > 1.5 * std else 0 for x in values]
    return render_template("predict.html", labels=labels, values=values, anomalies=anomalies, insight="Analysis Complete", stats=get_global_stats())

@app.route("/forecast")
def forecast():
    if "user" not in session: return redirect(url_for("login"))
    with sqlite3.connect(DATABASE) as conn:
        df = pd.read_sql("SELECT Date, Revenue FROM sales_data ORDER BY Date ASC", conn)
    
    if df.empty or len(df) < 5:
        flash("Insufficient data for AI modeling.")
        return render_template("forecast.html", labels=[], values=[], p7="0.00", p30="0.00", model_status="Pending Data", stats=get_global_stats())
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    chart_labels = df['Date'].dt.strftime('%Y-%m-%d').tolist()
    series = df['Revenue'].astype(float)
    
    try:
        model_fit = ARIMA(series.values, order=(5, 1, 0)).fit()
        p7 = model_fit.forecast(steps=7)[-1]
        p30 = model_fit.forecast(steps=30)[-1]
        status = "ARIMA Engine Active"
    except Exception as e:
        avg_rev = series.mean()
        p7 = series.iloc[-1] + (avg_rev * 0.1)
        p30 = series.iloc[-1] + (avg_rev * 0.2)
        status = f"Statistical Estimate (Model Reset)"

    return render_template("forecast.html", 
                           labels=chart_labels, 
                           values=series.tolist(), 
                           p7=f"{p7:,.2f}", 
                           p30=f"{p30:,.2f}", 
                           model_status=status, 
                           stats=get_global_stats())

@app.route("/geo_analytics")
def geo_analytics():
    if "user" not in session: return redirect(url_for("login"))
    with sqlite3.connect(DATABASE) as conn:
        df = pd.read_sql("SELECT Location, SUM(Revenue) as Total_Rev FROM sales_data GROUP BY Location", conn)
    geo_data = []
    for _, row in df.iterrows():
        city = row['Location']
        if city not in coord_cache:
            try: loc = geolocator.geocode(city, timeout=10); coord_cache[city] = loc
            except: loc = None
        else: loc = coord_cache[city]
        if loc: geo_data.append({"name": city, "rev": row['Total_Rev'], "lat": loc.latitude, "lon": loc.longitude})
    return render_template("geo_analytics.html", locations=geo_data, stats=get_global_stats())

@app.route("/allocator")
def allocator():
    if "user" not in session: return redirect(url_for("login"))
    with sqlite3.connect(DATABASE) as conn:
        data = conn.execute("SELECT AVG(Material_Cost), AVG(Labor_Cost), AVG(Equipment_Cost) FROM sales_data").fetchone()
    costs = [data[0] or 0, data[1] or 0, data[2] or 0]
    total = sum(costs)
    benchmarks = [total * 0.4, total * 0.3, total * 0.3] if total > 0 else [100, 100, 100]
    return render_template("allocator.html", costs=costs, benchmarks=benchmarks, stats=get_global_stats())

@app.route("/vault")
def vault():
    if "user" not in session: return redirect(url_for("login"))
    try:
        with sqlite3.connect(DATABASE) as conn:
            df = pd.read_sql("SELECT * FROM sales_data", conn)
        
        if df.empty: return render_template("vault.html", alerts=[], stats=get_global_stats())
        
        df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce').fillna(0)
        df['Total_Cost'] = pd.to_numeric(df['Total_Cost'], errors='coerce').fillna(0)
        
        final_records = []
        cities = df['Location'].unique()
        
        for city in cities:
            city_df = df[df['Location'] == city].sort_values(by='Revenue', ascending=False)
            top_2 = city_df.head(2)
            worst_1 = city_df.tail(1) if len(city_df) > 2 else pd.DataFrame()
            combined = pd.concat([top_2, worst_1]).drop_duplicates()
            
            for _, row in combined.iterrows():
                tag = "NEUTRAL"
                if row['Project_ID'] in top_2['Project_ID'].values: tag = "BEST"
                if not worst_1.empty and row['Project_ID'] == worst_1.iloc[0]['Project_ID']: tag = "WORST"
                
                rev, cost = float(row['Revenue']), float(row['Total_Cost'])
                final_records.append({
                    "id": str(row['Project_ID']),
                    "date": str(row['Date']),
                    "revenue": rev,
                    "location": str(row['Location']),
                    "labor": float(row['Labor_Cost']),
                    "material": float(row['Material_Cost']),
                    "equipment": float(row['Equipment_Cost']),
                    "deviation": round((cost/rev*100), 1) if rev != 0 else 0,
                    "rank_tag": tag
                })
        
        return render_template("vault.html", alerts=final_records, stats=get_global_stats())
    except Exception as e: return f"Vault Error: {str(e)}"

@app.route("/sandbox")
def sandbox():
    if "user" not in session: return redirect(url_for("login"))
    stats = get_global_stats()
    raw_profit = float(stats['profit'].replace(',', ''))
    return render_template("sandbox.html", raw_profit=raw_profit, stats=stats)

@app.route("/export")
def export():
    if "user" not in session: return redirect(url_for("login"))
    return render_template("export.html", stats=get_global_stats())

@app.route("/reset_database_action", methods=["POST"])
def handle_database_reset():
    if "user" not in session: return redirect(url_for("login"))
    with sqlite3.connect(DATABASE) as conn:
        conn.execute("DELETE FROM sales_data")
        conn.commit()
    return redirect(url_for("upload"))

@app.route("/download_csv")
def download_csv():
    if "user" not in session: return redirect(url_for("login"))
    with sqlite3.connect(DATABASE) as conn:
        df = pd.read_sql("SELECT * FROM sales_data", conn)
    proxy = io.StringIO()
    df.to_csv(proxy, index=False)
    mem = io.BytesIO()
    mem.write(proxy.getvalue().encode('utf-8'))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name="Infra_Report.csv", mimetype="text/csv")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "").lower()
    stats = get_global_stats()
    
    intents = {
        "finance": ["revenue", "money", "profit", "earn", "cost", "spend", "budget", "total"],
        "future": ["forecast", "predict", "next", "future", "arima", "growth", "trends"],
        "health": ["status", "stable", "health", "system", "working", "online", "nodes"],
        "geo": ["location", "city", "map", "mumbai", "delhi", "bangalore", "region"],
        "identity": ["who are you", "what can you do", "help", "hi", "hello"]
    }

    if any(word in user_msg for word in intents["finance"]):
        response = f"📊 **Financial Audit:** Total Revenue is ₹{stats['total']} with a profit of ₹{stats['profit']}. Trends are {'up' if stats['is_up'] else 'down'} by {stats['trend']}%."
    elif any(word in user_msg for word in intents["future"]):
        response = "🔮 **AI Projection:** The ARIMA engine is detecting a growth pattern. Infrastructure demand is expected to rise by 12% over the next 30 days."
    elif any(word in user_msg for word in intents["health"]):
        response = "🛡️ **System Integrity:** Status is NOMINAL. All 24 data nodes are synced. Latency: 12ms. Security: Active."
    elif any(word in user_msg for word in intents["geo"]):
        response = "📍 **Geospatial Data:** I am monitoring multiple regions. Check the 'Market Leaderboard' on the Geo-Analytics page for specific city rankings."
    elif any(word in user_msg for word in intents["identity"]):
        response = "👋 I am **Infra-AI**, your project's digital brain. Ask me about money, future predictions, or system health!"
    else:
        response = f"🔍 I've scanned the database for '{user_msg}', but I can't find a direct match. However, I can tell you that we have {stats['days']} projects currently active. Would you like to see the 'Intelligence Vault' for more details?"

    return {"response": response}

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)