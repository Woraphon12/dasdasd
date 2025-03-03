import pandas as pd
import numpy as np
import os

# ตั้งค่าจำนวนตัวอย่างข้อมูล
num_samples = 1000  

# สร้างข้อมูลสุ่มเสมือนค่าจากเซ็นเซอร์
np.random.seed(42)
data = {
    "Temperature": np.random.uniform(50, 120, num_samples).round(2),
    "Vibration": np.random.uniform(0.1, 1.5, num_samples).round(3),
    "Machine_Age": np.random.uniform(0, 10, num_samples).round(1),
    "Humidity": np.random.uniform(20, 80, num_samples).round(2),
    "RPM": np.random.uniform(800, 3000, num_samples).round(1),
    "Operating_Hours": np.random.uniform(100, 10000, num_samples).round(1),
}

# กำหนดเงื่อนไขการบำรุงรักษา (Maintenance_Required)
# ถ้าอุณหภูมิสูง, การสั่นสะเทือนสูง, และเครื่องเก่ามีแนวโน้มต้องซ่อมบำรุง
data["Maintenance_Required"] = np.where(
    (data["Temperature"] > 100) & (data["Vibration"] > 1.0) & (data["Machine_Age"] > 7),
    1,  # ต้องซ่อม
    0   # ไม่ต้องซ่อม
)

# สร้าง DataFrame
df = pd.DataFrame(data)

# กำหนดพาธที่ต้องการบันทึกไฟล์
folder_path = r"E:\uuuuuuuuuuuuuuuuuuu\data"
file_path = os.path.join(folder_path, "sensor_data_1000.csv")

# ตรวจสอบและสร้างโฟลเดอร์ ถ้ายังไม่มี
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# บันทึกไฟล์ CSV
df.to_csv(file_path, index=False, encoding='utf-8-sig')

print(f"✅ ไฟล์ถูกบันทึกที่: {file_path}")
