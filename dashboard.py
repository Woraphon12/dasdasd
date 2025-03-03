import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import random
import joblib
import time
import base64
from datetime import datetime

# โหลดโมเดล Machine Learning และ Label Encoder
loaded_model = joblib.load("machine_failure_model.pkl")
loaded_le = joblib.load("label_encoder.pkl")

# ฟังก์ชันจำลองค่าจากเซ็นเซอร์
def generate_sensor_data():
    return {
        "Temperature": round(random.uniform(50, 120), 2),  # °C
        "Vibration": round(random.uniform(0.1, 2.0), 2),  # G-force
        "Machine_Age": random.randint(1, 10),             # Years
        "Humidity": random.randint(30, 70),               # %
        "RPM": random.randint(1000, 5000),               # รอบต่อนาที
        "Operating_Hours": random.randint(1000, 8000)    # ชั่วโมง
    }

# ตั้งค่าเริ่มต้นของกราฟ
sensor_data_history = pd.DataFrame(columns=["Time", "Temperature", "Vibration", "RPM"])

# ค่าเริ่มต้นสำหรับการแสดงข้อมูลเซ็นเซอร์
last_sensor_data = {
    "Temperature": 0,
    "Vibration": 0,
    "Machine_Age": 0,
    "Humidity": 0,
    "RPM": 0,
    "Operating_Hours": 0
}

# สร้าง DataFrame สำหรับเก็บประวัติการตรวจพบความผิดปกติ
abnormality_history = pd.DataFrame(columns=["Timestamp", "Abnormality_Type", "Temperature", "Vibration", "RPM", "Humidity"])

# กำหนด External Stylesheets และ custom CSS
external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# สร้างโฟลเดอร์ assets ถ้ายังไม่มี
import os
if not os.path.exists("assets"):
    os.makedirs("assets")

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>AI Predictive Maintenance Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Roboto', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                color: #ffffff;
                background-color: #121212; /* fallback if image fails */
                background-image: url('/assets/background.jpg');
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
                min-height: 100vh;
            }
            
            /* Overlay to make background darker for better readability */
            body::before {
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.7); /* Dark overlay */
                z-index: -1;
            }
            
            .header {
                background: rgba(0, 0, 0, 0.7);
                color: #00bcd4;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                margin-bottom: 20px;
                border-bottom: 2px solid #00bcd4;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
            }
            
            .dashboard-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 20px 20px 20px;
            }
            
            .card {
                background-color: rgba(30, 30, 30, 0.8);
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
                padding: 20px;
                margin-bottom: 20px;
                border: 1px solid #333333;
                backdrop-filter: blur(5px);
                -webkit-backdrop-filter: blur(5px);
            }
            
            .card-title {
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 15px;
                color: #00bcd4;
                border-bottom: 1px solid #444444;
                padding-bottom: 8px;
            }
            
            .sensor-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            
            .sensor-value {
                display: flex;
                align-items: center;
                justify-content: space-between;
                background-color: rgba(40, 40, 40, 0.9);
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
                border-left: 3px solid #00bcd4;
                transition: all 0.3s ease;
            }
            
            .sensor-value:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0, 188, 212, 0.4);
            }
            
            .sensor-label {
                font-weight: 500;
                color: #bbbbbb;
            }
            
            .sensor-reading {
                font-size: 22px;
                font-weight: 600;
                color: #ffffff;
                text-shadow: 0 0 5px rgba(0, 188, 212, 0.5);
            }
            
            .sensor-icon {
                font-size: 24px;
                margin-right: 10px;
                color: #00bcd4;
            }
            
            .normal {
                background-color: rgba(27, 94, 32, 0.8);
                color: #e8f5e9;
                border: 1px solid #2e7d32;
            }
            
            .warning {
                background-color: rgba(230, 81, 0, 0.8);
                color: #fff8e1;
                border: 1px solid #ff8f00;
            }
            
            .danger {
                background-color: rgba(183, 28, 28, 0.8);
                color: #ffebee;
                border: 1px solid #c62828;
            }
            
            .status-indicator {
                font-size: 24px;
                font-weight: 600;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
                backdrop-filter: blur(5px);
                -webkit-backdrop-filter: blur(5px);
            }
            
            .status-icon {
                font-size: 28px;
                margin-right: 10px;
            }
            
            .info-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
            }
            
            .info-item {
                display: flex;
                flex-direction: column;
                padding: 10px;
                background-color: rgba(40, 40, 40, 0.9);
                border-radius: 5px;
            }
            
            @media (max-width: 600px) {
                .info-grid {
                    grid-template-columns: 1fr;
                }
            }
            
            /* สีเรืองแสงสำหรับข้อมูล */
            .glow-text {
                text-shadow: 0 0 5px #00bcd4, 0 0 10px rgba(0, 188, 212, 0.5);
            }
            
            /* เพิ่มเอฟเฟ็กต์เคลื่อนไหว */
            @keyframes pulse {
                0% {
                    box-shadow: 0 0 0 0 rgba(0, 188, 212, 0.4);
                }
                70% {
                    box-shadow: 0 0 0 10px rgba(0, 188, 212, 0);
                }
                100% {
                    box-shadow: 0 0 0 0 rgba(0, 188, 212, 0);
                }
            }
            
            .pulse {
                animation: pulse 2s infinite;
            }
            
            /* สไตล์สำหรับกราฟ */
            .plot-container {
                border-radius: 8px;
                overflow: hidden;
            }
            
            /* สไตล์สำหรับตารางประวัติ */
            .history-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }
            
            .history-table th {
                background-color: rgba(0, 188, 212, 0.2);
                color: #00bcd4;
                padding: 12px 10px;
                text-align: left;
                font-weight: 500;
                border-bottom: 2px solid #444;
            }
            
            .history-table td {
                padding: 12px 10px;
                border-bottom: 1px solid #444;
                color: #e0e0e0;
            }
            
            .history-table tr:hover {
                background-color: rgba(0, 188, 212, 0.1);
            }
            
            .abnormality-tag {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 14px;
                font-weight: 500;
                background-color: rgba(183, 28, 28, 0.8);
                color: #fff;
            }
            
            .history-empty {
                text-align: center;
                padding: 20px;
                color: #888;
                font-style: italic;
            }
            
            .history-container {
                max-height: 300px;
                overflow-y: auto;
                scrollbar-width: thin;
                scrollbar-color: #00bcd4 #2a2a2a;
            }
            
            .history-container::-webkit-scrollbar {
                width: 8px;
            }
            
            .history-container::-webkit-scrollbar-track {
                background: #2a2a2a;
                border-radius: 4px;
            }
            
            .history-container::-webkit-scrollbar-thumb {
                background-color: #00bcd4;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# สร้างไฟล์ HTML สำหรับแจ้งเตือนเพื่อวางรูปภาพพื้นหลัง
background_notice = """
<html>
<body>
    <h2>คำแนะนำการใช้รูปภาพพื้นหลัง</h2>
    <p>กรุณาจัดเตรียมรูปภาพพื้นหลังโดยทำตามขั้นตอนดังนี้:</p>
    <ol>
        <li>สร้างโฟลเดอร์ชื่อ "assets" ในโฟลเดอร์เดียวกับไฟล์ Python นี้</li>
        <li>วางรูปภาพลงในโฟลเดอร์ assets และตั้งชื่อเป็น "background.jpg"</li>
    </ol>
    <p>หากต้องการใช้ชื่อไฟล์อื่น ให้แก้ไขที่ <code>background-image: url('/assets/background.jpg');</code> ในโค้ด CSS</p>
</body>
</html>
"""

# บันทึกไฟล์แจ้งเตือน
with open("background_notice.html", "w", encoding="utf-8") as f:
    f.write(background_notice)

# ส่วนประกอบ UI ของแอป
app.layout = html.Div([
    # Header
    html.Div([
        html.H1([html.I(className="fas fa-cogs", style={"marginRight": "15px"}), "AI Predictive Maintenance Dashboard"], 
                style={"margin": "0", "textAlign": "center"})
    ], className="header"),
    
    # Main container
    html.Div([
        # Status indicator
        html.Div(id="prediction-output", className="status-indicator normal pulse"),
        
        # Sensor readings
        html.Div([
            html.Div([
                html.Div("Real-Time Sensor Readings", className="card-title")
            ]),
            html.Div(id="sensor-readings", className="sensor-grid")
        ], className="card"),
        
        # Graphs
        html.Div([
            html.Div("Sensor Data Trends", className="card-title"),
            dcc.Graph(id="sensor-graph", className="plot-container")
        ], className="card"),
        
        # ประวัติความผิดปกติ
        html.Div([
            html.Div("ประวัติการตรวจพบความผิดปกติ", className="card-title"),
            html.Div(id="abnormality-history", className="history-container")
        ], className="card"),
        
        # รายละเอียดเพิ่มเติม
        html.Div([
            html.Div("Machine Information", className="card-title"),
            html.Div([
                html.Div([
                    html.Div([
                        html.Div("Machine ID:", className="sensor-label"),
                        html.Div("M-7842", className="glow-text", style={"fontWeight": "600", "color": "#ffffff"})
                    ]),
                    html.Div([
                        html.Div("Location:", className="sensor-label"),
                        html.Div("Factory Line B - Station 3", style={"fontWeight": "600", "color": "#ffffff"})
                    ])
                ], className="info-item"),
                html.Div([
                    html.Div([
                        html.Div("Last Maintenance:", className="sensor-label"),
                        html.Div("2024-12-15", style={"fontWeight": "600", "color": "#ffffff"})
                    ]),
                    html.Div([
                        html.Div("Next Scheduled:", className="sensor-label"),
                        html.Div("2025-04-15", className="glow-text", style={"fontWeight": "600", "color": "#ffffff"})
                    ])
                ], className="info-item")
            ], className="info-grid")
        ], className="card"),
        
        # อัปเดตข้อมูลทุก 1 วินาที
        dcc.Interval(id="interval-update", interval=1000, n_intervals=0)
    ], className="dashboard-container")
])

# ฟังก์ชันอัปเดตข้อมูลแบบเรียลไทม์
@app.callback(
    [Output("sensor-graph", "figure"),
     Output("prediction-output", "children"),
     Output("prediction-output", "className"),
     Output("sensor-readings", "children"),
     Output("abnormality-history", "children")],
    [Input("interval-update", "n_intervals")]
)
def update_dashboard(n):
    global sensor_data_history, last_sensor_data, abnormality_history
    
    # รับค่าจากเซ็นเซอร์ (จำลอง)
    new_data = generate_sensor_data()
    last_sensor_data = new_data  # เก็บค่าล่าสุดไว้
    timestamp = time.strftime("%H:%M:%S")  # เวลา
    full_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # วันที่และเวลาเต็มรูปแบบ

    # สร้าง DataFrame สำหรับทำนาย
    new_df = pd.DataFrame([new_data])
    predicted_numeric = loaded_model.predict(new_df)
    predicted_abnormality = loaded_le.inverse_transform(predicted_numeric)[0]

    # เพิ่มข้อมูลใหม่ลงใน DataFrame
    new_entry = pd.DataFrame({"Time": [timestamp], 
                              "Temperature": [new_data["Temperature"]],
                              "Vibration": [new_data["Vibration"]],
                              "RPM": [new_data["RPM"]]})
    
    # แก้ไขเพื่อหลีกเลี่ยง FutureWarning
    if sensor_data_history.empty:
        sensor_data_history = new_entry
    else:
        sensor_data_history = pd.concat([sensor_data_history, new_entry], ignore_index=True)
        
    # จำกัดขนาดให้เก็บแค่ 30 จุดล่าสุด
    if len(sensor_data_history) > 30:
        sensor_data_history = sensor_data_history.tail(30)

    # บันทึกประวัติเมื่อตรวจพบความผิดปกติ
    # กำหนดให้มีโอกาส 1/15 ที่จะตรวจพบความผิดปกติ (สำหรับการสาธิต)
    if predicted_abnormality != "Normal" or random.randint(1, 15) == 1:
        # แบบสุ่ม: ถ้าการทำนายเป็น Normal และสุ่มได้ให้สร้างความผิดปกติจำลอง
        if predicted_abnormality == "Normal":
            failure_types = ["Bearing Failure", "Motor Overheating", "Misalignment", "Loose Components", "Excessive Load"]
            predicted_abnormality = random.choice(failure_types)
        
        # บันทึกข้อมูลความผิดปกติ
        abnormality_entry = pd.DataFrame({
            "Timestamp": [full_timestamp],
            "Abnormality_Type": [predicted_abnormality],
            "Temperature": [new_data["Temperature"]],
            "Vibration": [new_data["Vibration"]],
            "RPM": [new_data["RPM"]],
            "Humidity": [new_data["Humidity"]]
        })
        
        if abnormality_history.empty:
            abnormality_history = abnormality_entry
        else:
            abnormality_history = pd.concat([abnormality_history, abnormality_entry], ignore_index=True)
        
        # จำกัดประวัติให้แสดงแค่ 20 รายการล่าสุด
        if len(abnormality_history) > 20:
            abnormality_history = abnormality_history.tail(20)

    # สร้างกราฟ
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sensor_data_history["Time"], y=sensor_data_history["Temperature"], 
                             mode="lines+markers", name="Temperature (°C)", line=dict(color="#ff5722", width=3)))
    fig.add_trace(go.Scatter(x=sensor_data_history["Time"], y=sensor_data_history["Vibration"], 
                             mode="lines+markers", name="Vibration (G-force)", line=dict(color="#00bcd4", width=3)))
    fig.add_trace(go.Scatter(x=sensor_data_history["Time"], y=sensor_data_history["RPM"], 
                             mode="lines+markers", name="RPM", line=dict(color="#76ff03", width=3)))

    fig.update_layout(
        title=None,
        xaxis_title="Time",
        yaxis_title="Value",
        plot_bgcolor="rgba(30, 30, 30, 0.8)",
        paper_bgcolor="rgba(30, 30, 30, 0)",
        font=dict(family="Roboto, sans-serif", color="#e0e0e0"),
        legend=dict(orientation="h", y=1.1, font=dict(color="#e0e0e0"), bgcolor="rgba(30, 30, 30, 0.7)"),
        margin=dict(l=40, r=40, t=20, b=40),
        xaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)", tickfont=dict(color="#e0e0e0")),
        yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)", tickfont=dict(color="#e0e0e0")),
        hovermode="x unified"
    )

    # กำหนดสถานะและไอคอน
    status_class = "status-indicator normal pulse" if predicted_abnormality == "Normal" else "status-indicator danger pulse"
    status_icon = html.I(className="fas fa-check-circle status-icon") if predicted_abnormality == "Normal" else html.I(className="fas fa-exclamation-triangle status-icon")
    
    # แสดงผลการทำนาย
    prediction_text = [
        status_icon,
        "ระบบทำงานปกติ" if predicted_abnormality == "Normal" else f"ตรวจพบความผิดปกติ: {predicted_abnormality}"
    ]
    
    # สร้าง UI สำหรับแสดงค่าเซ็นเซอร์
    sensor_readings = [
        html.Div([
            html.Div([html.I(className="fas fa-thermometer-half sensor-icon"), "อุณหภูมิ"], className="sensor-label"),
            html.Div(f"{new_data['Temperature']} °C", className="sensor-reading")
        ], className="sensor-value"),
        
        html.Div([
            html.Div([html.I(className="fas fa-vibration sensor-icon"), "แรงสั่นสะเทือน"], className="sensor-label"),
            html.Div(f"{new_data['Vibration']} G", className="sensor-reading")
        ], className="sensor-value"),
        
        html.Div([
            html.Div([html.I(className="fas fa-tachometer-alt sensor-icon"), "RPM"], className="sensor-label"),
            html.Div(f"{new_data['RPM']}", className="sensor-reading")
        ], className="sensor-value"),
        
        html.Div([
            html.Div([html.I(className="fas fa-tint sensor-icon"), "ความชื้น"], className="sensor-label"),
            html.Div(f"{new_data['Humidity']}%", className="sensor-reading")
        ], className="sensor-value"),
        
        html.Div([
            html.Div([html.I(className="fas fa-calendar-alt sensor-icon"), "อายุเครื่อง"], className="sensor-label"),
            html.Div(f"{new_data['Machine_Age']} ปี", className="sensor-reading")
        ], className="sensor-value"),
        
        html.Div([
            html.Div([html.I(className="fas fa-clock sensor-icon"), "ชั่วโมงทำงาน"], className="sensor-label"),
            html.Div(f"{new_data['Operating_Hours']} ชม.", className="sensor-reading")
        ], className="sensor-value"),
    ]
    
    # สร้างตารางแสดงประวัติความผิดปกติ
    if abnormality_history.empty:
        abnormality_table = html.Div("ยังไม่มีประวัติการตรวจพบความผิดปกติ", className="history-empty")
    else:
        # เรียงข้อมูลจากใหม่ไปเก่า
        sorted_history = abnormality_history.sort_values(by="Timestamp", ascending=False)
        
        # สร้างตาราง
        abnormality_table = html.Table([
            html.Thead(
                html.Tr([
                    html.Th("เวลา"),
                    html.Th("ประเภทความผิดปกติ"),
                    html.Th("อุณหภูมิ (°C)"),
                    html.Th("แรงสั่นสะเทือน (G)"),
                    html.Th("RPM"),
                    html.Th("ความชื้น (%)")
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(row["Timestamp"]),
                    html.Td(html.Span(row["Abnormality_Type"], className="abnormality-tag")),
                    html.Td(f"{row['Temperature']:.2f}"),
                    html.Td(f"{row['Vibration']:.2f}"),
                    html.Td(f"{row['RPM']}"),
                    html.Td(f"{row['Humidity']}"),
                ]) for _, row in sorted_history.iterrows()
            ])
        ], className="history-table")

    return fig, prediction_text, status_class, sensor_readings, abnormality_table

# รันแอป
if __name__ == "__main__":
    app.run_server(debug=True)