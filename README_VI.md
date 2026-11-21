# NTN LEO Research Lab — Step 2 sửa độ phủ + Step 3 LSTM

## Mục đích bản cập nhật

Bản này giữ nguyên trang mô phỏng Trái Đất 3D, texture bản đồ, lưới kinh/vĩ tuyến, trạm mặt đất, quỹ đạo Kepler, chuyển ECI → ECEF và link budget Ka-band trong `app.py` bạn gửi.

Điểm sửa chính nằm ở bộ tạo Dataset:

- Bản cũ dịch RAAN trung tâm thêm 11° sau mỗi kịch bản và pha thêm 23° một cách cố định. Với UE cố định, nhiều cấu hình bị lệch khỏi vùng phủ của Hà Nội; CSV trước đó chỉ có SCN_01 có candidate.
- Bản mới đánh giá nhanh các tổ hợp RAAN/pha trước khi tạo dữ liệu. Chương trình ưu tiên chọn kịch bản có ít nhất một khoảng thời gian liên kết đủ điều kiện tại UE.
- Vẫn lưu tất cả vệ tinh và mọi thời điểm, kể cả dưới chân trời; không xóa các mẫu outage.
- Bổ sung bảng audit theo từng scenario để thấy số mẫu candidate và Elevation cực đại.
- Bổ sung `ai/train_lstm.py` để huấn luyện LSTM, chia train/validation/test theo `scenario_id`, so sánh với baseline giữ nguyên SNR cuối cùng, và lưu MAE/RMSE.

## A. Cài và chạy website

1. Giải nén toàn bộ ZIP vào thư mục mới, ví dụ `C:\NTN_AI_STEP2_LSTM`.
2. Không xóa `bmng_preview.jpg`, `requirements.txt` hoặc các file trong ZIP.
3. Chạy `CHAY_UNG_DUNG.bat` hoặc mở Terminal tại thư mục dự án và chạy:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

4. Mở `http://localhost:8501`.
5. Chọn trang `02 — Time-series Link Quality`.

## B. Sinh lại Dataset — không dùng CSV cũ để huấn luyện chính thức

Các CSV đã tải ở phiên bản trước có 10 scenario nhưng chỉ SCN_01 có candidate; các scenario còn lại không có mẫu đạt Elevation ≥ 10°. Dữ liệu đó không phù hợp để đánh giá AI lựa chọn vệ tinh theo nhiều kịch bản.

Dùng cấu hình kiểm tra ban đầu:

| Tham số | Giá trị |
|---|---:|
| UE | TĐ. Hà Nội |
| Số mặt phẳng | 3 |
| Vệ tinh/mặt phẳng | 4 |
| Độ cao | 600 km |
| Góc nghiêng | 53° |
| RAAN trung tâm | 98° |
| Khoảng cách RAAN giữa mặt phẳng | 4° |
| Số kịch bản | 10 |
| Thời gian mỗi kịch bản | 720 s |
| Chu kỳ lấy mẫu | 1 s |

Nhấn `GENERATE DATASET`. Sau khi chạy xong, kiểm tra bảng `Kiểm tra độ phủ theo từng kịch bản`:

- Phải có 10/10 scenario có `Mẫu đủ điều kiện` lớn hơn 0.
- Với cấu hình trên, số dòng dự kiến là `10 × 721 × 12 = 86.520`.
- Giá trị thiếu và NaN/vô cực phải bằng 0.

Tải CSV và đặt đúng tên, đúng vị trí:

`data\ntn_link_quality_dataset.csv`

Nếu trình duyệt tải về tên `ntn_link_quality_dataset (1).csv`, hãy đổi tên thành `ntn_link_quality_dataset.csv` trước khi chép vào `data`.

## C. Huấn luyện LSTM

Khuyến nghị dùng Python 3.12 trên Windows. Cần Internet lần đầu để cài PyTorch và các thư viện.

Cách đơn giản:

1. Đóng website hoặc để website không chạy trong lúc huấn luyện.
2. Đặt CSV vào `data\ntn_link_quality_dataset.csv`.
3. Nhấp đúp `CHAY_HUAN_LUYEN_AI.bat`.

Hoặc chạy trong Terminal:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements_ai.txt
.venv\Scripts\python.exe ai\train_lstm.py --csv data\ntn_link_quality_dataset.csv --output-dir results\step3_lstm
```

Cấu hình mặc định:

- Input: 30 mẫu quá khứ.
- Horizon: dự báo SNR sau 5 giây.
- Features: elevation, distance, range rate, Doppler, delay, FSPL, công suất thu, SNR hiện tại và trạng thái trên chân trời.
- Mô hình: LSTM 1 lớp, 32 hidden units, Dense head.
- Tối đa 15 epoch, batch size 1024, lấy cửa sổ cách nhau 5 mẫu để giảm thời gian chạy CPU; early stopping.
- Chia train/validation/test theo kịch bản, không chia ngẫu nhiên từng dòng.

## D. Kết quả sau huấn luyện

Mô hình và scaler:

- `models\lstm_snr.pth`
- `models\lstm_scalers.joblib`

Kết quả đánh giá:

- `results\step3_lstm\metrics.json`
- `results\step3_lstm\scenario_split.json`
- `results\step3_lstm\training_history.csv`
- `results\step3_lstm\test_predictions.csv`
- `results\step3_lstm\training_loss.png`
- `results\step3_lstm\prediction_vs_actual.png`
- `results\step3_lstm\error_histogram.png`

Script báo MAE/RMSE của LSTM và baseline persistence. Không kết luận LSTM tốt hơn nếu nó không vượt baseline trên test set.

## Ghi chú mô hình

Đây là mô hình nghiên cứu dùng quỹ đạo tròn Kepler và link budget giả định. Nó chưa dùng TLE/SGP4, chưa có mô hình fading/mưa chi tiết và chưa đại diện cho hệ thống 3GPP hoàn chỉnh. Kết quả AI chỉ có ý nghĩa trong phạm vi giả thiết và Dataset đã tạo.
