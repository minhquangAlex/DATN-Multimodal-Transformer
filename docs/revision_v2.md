# Biên bản sửa và kiểm chứng v2 — 16/09/2026

## Kết quả đã hoàn thành

| Vấn đề trước đây | Sửa trong v2 | File chính |
|---|---|---|
| Chọn sản phẩm từ toàn kỳ, nhìn dữ liệu tương lai | Tiêu chí và sampling chỉ dùng train | src/data/select_articles.py |
| Hai cách chia thời gian khác nhau | Một cấu hình cutoff theo ngày; loader kiểm tra split được lưu | src/config.py, src/data/splits.py, src/data/dataset.py |
| 106 tuần có tuần biên thiếu ngày; tài liệu ghi 108 | Bỏ hai tuần không đầy đủ, còn 104; tạo lại dữ liệu | src/data/build_weekly_sales.py |
| Sampling làm tròn từng nhóm, không đủ đúng cỡ mẫu | Phân bổ quota chính xác, giữ nhóm nhỏ khi đủ ngân sách | src/data/select_articles.py |
| Ghi đè bảng tuần sau sampling | Tách weekly_sales_all và weekly_sales_selected | src/data/select_articles.py |
| Đọc toàn bộ CSV giao dịch vào RAM | EDA và tổng hợp theo chunk, chỉ dùng cột cần thiết | src/data/eda.py, src/data/build_weekly_sales.py |
| Hàng trăm nghìn dictionary cửa sổ | Một ma trận và danh sách origin, tạo mẫu khi truy cập | src/data/dataset.py |
| Ảnh lỗi thành ảnh đen rồi encode | Audit ảnh thật; missing placeholder và availability riêng | src/features/extract_embeddings.py |
| Vector ảnh cũ collapse không bị phát hiện | Kiểm tra phương sai, vector phân biệt, zero và mọi shape | src/features/verify_embeddings.py |
| ID embedding và CSV lệch; metadata vẫn passed | Exact key matching, hash data/embedding/availability, cập nhật failed khi kiểm định lỗi | src/features/verify_embeddings.py |
| Text được ghép khác giữa local/Kaggle | Dùng đúng text_for_embedding của Stage 1 | src/features/extract_embeddings.py |
| Seasonal naive thực chất last-value | Đổi tên last-value; thêm seasonal naive thật có kiểm tra 52 tuần | src/models/baselines.py |
| Linear baseline thực chất MLP | Tách Linear một lớp và MLP | src/models/baselines.py |
| Scale giữa sales/image/text chênh lệch | Encoder L2-normalized; tùy chọn log1p nhất quán cho mô hình học | src/features/extract_embeddings.py, src/utils/trainer.py |
| Chưa kiểm soát random seed và chưa lưu đủ thực nghiệm | 3 seed mặc định, metadata/versions/hash, checkpoint và lịch sử train | src/utils/reproducibility.py, src/pipeline/train_baselines.py |
| Missing ablation chỉ train lại | Tách same-checkpoint và retrained, thêm partial history test | src/pipeline/train_baselines.py |
| wMAPE đơn vị khác sMAPE và nhóm zero không rõ nghĩa | Nhãn % nhất quán, nhóm tổng actual=0 để undefined | src/utils/metrics.py |
| Chỉ lưu metric tổng | Lưu dự đoán, metric horizon/nhóm/độ thưa, seed summary và mức suy giảm | src/pipeline/train_baselines.py |
| Notebook giả định đường dẫn local trên Kaggle | Bootstrap có override, từ chối input nhập nhằng, output vào working | notebooks/, src/utils/paths.py |
| Tải nhầm embedding cũ từ Input | Archive từ output vừa trích xuất | notebooks/02_MultiModal_Vector.ipynb |

## Dữ liệu thật đã dựng lại

Đã đọc toàn bộ 31.788.324 giao dịch. Bộ mới ở `processed_data/v2/` có 8.000 sản phẩm lấy từ 33.514 ứng viên train-only; 104 tuần đầy đủ; 832.000 dòng; tỷ lệ zero 60,1577%. Train/validation/test là 80/12/12 tuần.

Với L=12 và H=2: 536.000 mẫu train, 88.000 validation, 88.000 test. Metadata có cutoff, thống kê và SHA-256. Bảng tuần toàn bộ sản phẩm được giữ để chạy lại sampling mà không cần tổng hợp CSV thô lần nữa.

Ba baseline deterministic đã chạy trên toàn bộ test thật. Bảng và dự đoán nằm tại `results/v2/20260916T150005Z_ea67e16e/`; kết quả được ghi trong tài liệu Stage 3. Mỗi run lưu hash source tại thời điểm chạy, vì vậy có thể truy vết ngay cả khi tài liệu hoặc chức năng kiểm định được cải thiện sau run đó.

## Kiểm chứng mã nguồn

Lệnh kiểm thử:

```bash
python -m pytest -q -p no:cacheprovider
```

**19 kiểm thử đã qua**, bao gồm:

- ISO week qua ranh giới năm và loại tuần biên thiếu ngày.
- Sampling không dùng sản phẩm chỉ xuất hiện tương lai; thống kê chỉ thuộc train.
- Cỡ mẫu chính xác, nhóm nhỏ được giữ, bảng tổng hợp không bị thu hẹp sau khi chạy lại.
- Target không vượt ranh giới split; validation vẫn có lịch sử train hợp lệ.
- Chặn index bị thay đổi hash và cấu hình không tạo được cửa sổ.
- Chặn embedding collapse, shape sai ở vector không phải vector đầu, provenance lệch.
- Availability tự nhiên, zero placeholder, chế độ full và trạng thái failed.
- Audit ảnh thật, ảnh thiếu và ảnh hỏng.
- Luồng extraction dùng **encoder giả trong test** trên ảnh màu thử nghiệm: xuất file, chuẩn hóa và kiểm định; không tải encoder thật.
- Mask tác động đúng vào input, không sửa batch gốc, không thay trọng số khi evaluate.
- Naive/seasonal naive, metric zero và kiểm tra shape, transform/inverse transform.
- Luồng train/evaluate/checkpoint/CSV cho Linear, MLP, LSTM, Transformer và multimodal MLP trên fixture nhỏ; hai protocol missing riêng biệt.
- EDA theo chunk và đường dẫn input nhập nhằng.

Ba notebook đã được kiểm tra theo schema nbformat, biên dịch mọi code cell và chạy bootstrap đường dẫn local. Không tuyên bố đã chạy toàn bộ notebook extraction/training trên Kaggle.

Windows chặn thư mục tạm pytest trong môi trường hạn chế ở lần chạy đầu; kiểm thử sau đó được chạy với quyền đã được cho phép. Đây là lỗi quyền thư mục tạm, không phải kiểm thử thuật toán thất bại.

## Đóng gói và tương thích

`artifacts/kaggle/source-v2.zip` chứa source mới; `artifacts/kaggle/hm-stage1-v2.zip` chứa ba file Stage 1 cần cho notebook 02–03. Manifest của gói ghi hash để đối chiếu. Các archive cũ, tạm thời và trùng lặp đã được loại khỏi workspace để tránh upload nhầm.

Dữ liệu cũ, embedding cũ và checkpoint cũ được giữ để đối chiếu. Metadata embedding cũ được đánh dấu failed và có báo cáo riêng; không xem chúng là đầu ra v2. Các notebook cũ được thay bằng notebook gọi mã chung, xóa output cũ để không hiển thị số liệu lạc phiên bản.

## Chưa hoàn thành và bước tiếp theo

- Chưa trích xuất CLIP/MiniLM thật cho danh sách v2. Máy local không có bộ ảnh H&M cần thiết; môi trường hiện tại cũng chưa có sentence-transformers. Đã có notebook và requirements để chạy trên Kaggle.
- Chưa huấn luyện toàn bộ learning baselines với ba seed trên dữ liệu thật. Test fixture chỉ xác nhận luồng thực thi, không đánh giá chất lượng nghiên cứu.
- Chưa xây Multimodal Transformer, learnable mask token, modality dropout hoặc contrastive alignment. Đây là Stage 4–5, không được mô tả như đã có trong v2.
- Calendar và availability đã có trong batch; baseline hiện chưa tận dụng calendar hoặc học từ mask token.
- Chưa bổ sung đánh giá partial-history cho từng baseline sales-only; cần đối chứng đó khi muốn kết luận về lợi ích đa phương thức trong cùng điều kiện lịch sử bị cắt.
- Chưa thể upload hay chạy trong tài khoản Kaggle từ workspace hiện tại. Các gói upload và hướng dẫn đã chuẩn bị ở `docs/kaggle_workflow.md`.

Ưu tiên tiếp theo: upload hai gói v2 → notebook 02 → kiểm định embedding → notebook 03 → phân tích kết quả đa seed trước khi xây Stage 4.


## Bổ sung minh họa và workload

Đã bổ sung workflow Mermaid vào tài liệu Stage 1–3, hình phân chia thời gian, phân bố nhóm sản phẩm và kết quả baseline thực tế. Tài liệu `research_workflow.md` mô tả sáu giai đoạn, workload chạy Kaggle, ma trận thực nghiệm và kiến trúc token-level fusion dự kiến cho Stage 4–5. Các phần dự kiến được ghi rõ là chưa triển khai.

Hình có cả PNG và SVG trong `docs/figures/`, cùng metadata nguồn. Đã kiểm tra trực quan ba hình PNG và kiểm tra đường dẫn hình trong Markdown. Lần bổ sung tài liệu này không thay đổi mã huấn luyện hoặc chạy lại benchmark; 19 kiểm thử được ghi ở trên thuộc đợt sửa pipeline v2 trước đó.
