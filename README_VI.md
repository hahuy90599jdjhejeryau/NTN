# NTN AI LEO — Bản vá hiệu năng và ổn định

Bản này được cập nhật từ `NTN_AI_LEO_FAST_SIM.zip` hiện có. Mục tiêu là xử lý các điểm được rà soát mà không đổi mô hình vật lý, mô hình 7 Beam, mô hình LSTM, ba chính sách Handover hoặc công thức KPI.

## Các chỉnh sửa trong bản này

1. **Xác nhận file hoàn chỉnh:** `app.py` có đầy đủ phần cuối của Trang 03 và đã được kiểm tra cú pháp bằng `py_compile`.
2. **Chọn max nhanh, ổn định:** thay `np.argsort(...)[0]` bằng `np.argmax(scores)`. Khi có điểm bằng nhau, `argmax` chọn phần tử đầu tiên theo thứ tự ứng viên; không cần sắp xếp toàn bộ.
3. **Không giữ KPI cũ sau khi áp dụng cấu hình mới:** khi nhấn `ÁP DỤNG & LƯU CẤU HÌNH`, kết quả Handover cũ trong `session_state` bị xóa. Trang 03 yêu cầu chạy lại với cấu hình mới.
4. **Chống lỗi `ho_duration`:** nếu `settings.json` chứa thời lượng không thuộc danh sách `[300, 600, 720, 1200]`, giao diện sẽ chọn mặc định 720 giây thay vì gây `ValueError`.
5. **Tối ưu Trang 02:** thay vòng lặp Python theo từng giây × từng vệ tinh bằng cùng hàm mô phỏng NumPy vector hóa đang dùng ở Trang 03. Tùy chọn lấy mẫu 1/2/5/10 giây vẫn được tôn trọng.
6. **Cache tìm kịch bản phủ sóng:** cấu hình pha/RAAN đã tìm được được cache theo các tham số và vị trí UE.

## Kiểm tra đã thực hiện

- `python -m py_compile app.py`: đạt.
- Đối chiếu mô phỏng scalar với mô phỏng vector hóa trên 3 mặt phẳng × 4 vệ tinh × 121 mốc thời gian: Beam ID và `candidate` trùng nhau.
- Sai khác cực đại đo được: Elevation `3.84e-13°`; khoảng cách `1.82e-12 km`; range-rate `1.78e-15 km/s`; Doppler `2.27e-13 kHz`; Beam offset `4.05e-14 dB`.
- Kiểm tra chu kỳ lấy mẫu 5 giây: các mốc `0, 5, 10, 15, 20` được tạo đúng.

Các kiểm tra trên xác nhận tương đương số học của mô hình trong các trường hợp thử nghiệm; chưa phải kiểm thử giao diện toàn bộ trên máy Windows của bạn.

## Cách cập nhật an toàn

**Chỉ thay file `app.py` trong thư mục dự án hiện tại. Không xóa toàn bộ thư mục dự án.**

1. Dừng Streamlit bằng `Ctrl + C` trong Terminal.
2. Sao lưu file cũ thành `app_backup_before_patch.py`.
3. Chép `app.py` trong gói này vào đúng thư mục dự án đang chạy, ghi đè file cũ.
4. Giữ nguyên các thư mục và file riêng của bạn:
   - `models/lstm_snr.pth`
   - `models/lstm_scalers.joblib`
   - `results/step3_lstm/`
   - `earth_texture.jpg` hoặc `earth_atmos_2048.jpg` hoặc `bmng_preview.jpg`
   - `settings.json`
   - `data/` và các CSV đã tạo.
5. Chạy lại:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

Mở `http://localhost:8502`.

## Lưu ý về tốc độ

- Trang 03: lần đầu với một cấu hình vật lý mới phải tính mô phỏng và dự báo; những lần sau với đúng cấu hình đó có thể dùng cache.
- Khi chỉ đổi Hysteresis, TTT hoặc AI Weight rồi nhấn `CHẠY MÔ PHỎNG`, kết quả vật lý và dự báo LSTM được tái sử dụng; phần chính sách có tham số thay đổi sẽ được tính lại.
- Trang 02: mô phỏng mỗi kịch bản đã vector hóa, và mỗi cấu hình kịch bản được cache. Lần đầu tạo một bộ Dataset mới vẫn tốn thời gian; lần tạo lại với đúng cấu hình có thể sử dụng cache.
- Cache Streamlit nằm trong bộ nhớ của tiến trình. Khi tắt hẳn ứng dụng, cache có thể phải tạo lại; `settings.json`, model và CSV vẫn nằm trên ổ đĩa.
- Thời gian thực tế còn phụ thuộc CPU, RAM, số vệ tinh, thời lượng, chu kỳ lấy mẫu, LSTM và việc dựng biểu đồ Plotly.
