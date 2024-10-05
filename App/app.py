from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Tải mô hình tốt nhất và LabelEncoder
best_model = joblib.load('E:/WebDuDoanDiemThi/App/Model/best_model.joblib')
label_encoder = joblib.load('E:/WebDuDoanDiemThi/App/Model/label_encoder.joblib')


@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_career = None
    highest_combination = None
    highest_value = 0
    if request.method == 'POST':
        # Nhận điểm từ form
        toan = float(request.form.get('toan', 0))
        van = float(request.form.get('van', 0))
        ngoaingu = float(request.form.get('ngoaingu', 0))
        vatly = float(request.form.get('vatly', 0))
        hoahoc = float(request.form.get('hoahoc', 0))
        sinhhoc = float(request.form.get('sinhhoc', 0))
        lichsu = float(request.form.get('lichsu', 0))
        dialy = float(request.form.get('dialy', 0))
        gdcd = float(request.form.get('gdcd', 0))

        # Tính toán điểm tổ hợp
        A = toan + vatly + hoahoc
        A01 = toan + vatly + ngoaingu
        B = toan + sinhhoc + hoahoc
        C = van + lichsu + dialy
        D = toan + van + ngoaingu

        # Tạo một danh sách các tổ hợp
        combinations = {'A': A, 'A01': A01, 'B': B, 'C': C, 'D': D}
        
        # Tìm tổ hợp cao nhất
        highest_combination = max(combinations, key=combinations.get)
        highest_value = combinations[highest_combination]

        # Dự đoán nghề nghiệp
        features = np.array([[A, A01, B, C, D]])
        predicted_class = best_model.predict(features)[0]
        predicted_career = label_encoder.inverse_transform([predicted_class])[0]

        return render_template('index.html', a=A, a01=A01, b=B, c=C, d=D, 
                               predicted_career=predicted_career, 
                               highest_combination=highest_combination, 
                               highest_value=highest_value)

    return render_template('index.html', a=None, a01=None, b=None, c=None, d=None, 
                           predicted_career=predicted_career, 
                           highest_combination=None, 
                           highest_value=None)

if __name__ == '__main__':
    app.run(debug=True)
