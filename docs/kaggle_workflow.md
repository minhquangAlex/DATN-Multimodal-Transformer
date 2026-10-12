# Làm việc với VS Code và Kaggle — v2

## Bộ file để đưa lên Kaggle

Trong workspace đã chuẩn bị:

- `artifacts/kaggle/source-v2.zip`: mã trong `src/`, notebook, requirements, tests và tài liệu v2.
- `artifacts/kaggle/hm-stage1-v2.zip`: dữ liệu Stage 1 mới, bao gồm index, selected articles, lịch tuần, bảng tổng hợp và metadata.
- `src.zip`: được cập nhật cùng mã nguồn mới để tương thích cách upload hiện tại.

Chỉ dùng hai archive hiện hành trong `artifacts/kaggle/`. Archive cũ và tạm đã được loại khỏi workspace để tránh upload nhầm hoặc tạo input nhập nhằng.

## 1. Tạo dataset input mới

Upload archive source thành dataset source v2 và archive Stage 1 thành dataset H&M Stage 1 v2. Giữ nguyên cấu trúc trong archive. Input cho extraction phải có `selected_articles.csv` từ v2; input cho training phải có cả `dataset_index.csv` và `stage1_summary.json`.

Đã dựng Stage 1 local nên không bắt buộc chạy notebook 01 lại trên Kaggle. Không sửa selected CSV hoặc index bằng Excel sau khi đóng gói vì sẽ thay đổi hash.

## 2. Chạy notebook 02 để tạo embedding

Gắn source v2, Stage 1 v2, H&M competition. Bật GPU và Internet khi cần tải encoder. Dùng `notebooks/02_MultiModal_Vector.ipynb` mới.

Notebook tự phát hiện thư mục khi chỉ có một bộ phù hợp. Nếu có nhiều bộ, đặt `CODE_ROOT_OVERRIDE`, `DATA_DIR_OVERRIDE`, `RAW_DATA_OVERRIDE` và/hoặc `IMAGES_DIR_OVERRIDE` bằng đường dẫn thực tế trong Input. Không dùng giả định `../raw_data` hay lấy file đầu tiên tìm được.

Nếu môi trường thiếu dependencies, có thể chạy một cell cài đặt sau khi đã xác định CODE_ROOT:

```python
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(CODE_ROOT/'requirements-extraction.txt')])
```

Nếu đã import phiên bản thư viện khác trước khi cài, khởi động lại session/kernel trước khi chạy tiếp. Các requirements định nghĩa khoảng phiên bản tương thích; metadata ghi phiên bản thực tế và revision encoder để tái lập lần chạy.

Xem 10 ảnh mẫu, kiểm tra audit rồi chạy extraction. Mặc định gặp ảnh thiếu/hỏng thì dừng để xử lý; không có fallback encode ảnh đen. Không tăng ngưỡng thiếu để che một đường dẫn ảnh sai.

Tải **hm-embeddings-v2.zip** từ output vừa tạo. Archive phải chứa `image/`, `text/`, `metadata.json`, `availability.csv`, `image_audit.json`, `validation_report.json`.

## 3. Chạy notebook 03

Upload embedding mới thành dataset input mới, gắn cùng source v2 và Stage 1 v2. Mở `notebooks/03_baselines.ipynb`.

Mặc định chạy 8 baseline, 3 seed cho các mô hình học, 30 epoch tối đa, patience 5. Bật retrained ablation để có thêm bảng huấn luyện lại từng tổ hợp. Việc dùng cùng full checkpoint dưới các tình huống missing đã được thực hiện tự động cho multimodal MLP.

Để thử trước khi có embedding mới:

```python
MODELS = ['zero', 'last_value', 'moving_average', 'linear', 'mlp', 'lstm', 'transformer']
RETRAINED_ABLATION = False
```

Đặt hai giá trị ngay trong cell cấu hình trước khi chạy các dòng giải quyết đường dẫn. Chế độ sales-only không nạp embedding.

Kết quả nằm trong `/kaggle/working/results/v2/<run_id>/`. Download archive ở cell cuối, bao gồm config, checkpoint, dự đoán và các bảng đánh giá.

## 4. Đưa kết quả về VS Code

- Giải nén embedding vào `embeddings/v2/` sao cho đường dẫn cuối là `embeddings/v2/image/image_embeddings.pt` và `embeddings/v2/metadata.json`.
- Giữ kết quả từng run trong một thư mục riêng; không trộn checkpoint/CSV giữa các seed hoặc phiên bản dữ liệu.
- Chạy `python -m src.features.verify_embeddings` để kiểm định lại ở local.
- Nếu đổi danh sách sản phẩm, nội dung text hoặc cách chia dữ liệu: tạo lại các artifact phụ thuộc và một run mới.

## Phạm vi tự động hóa

Mã, dữ liệu v2, kiểm thử và archive đã được chuẩn bị local. Việc upload dataset, bật GPU, chấp nhận điều khoản competition nếu tài khoản yêu cầu và chạy notebook trên tài khoản Kaggle do người dùng thực hiện. Chưa có kết quả trích xuất ảnh thật v2 hoặc benchmark deep learning đầy đủ trong lần sửa này.
