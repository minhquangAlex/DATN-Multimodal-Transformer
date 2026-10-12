# Cài bộ skill cho đồ án

Ba thư mục cùng cấp trong gói này là ba skill độc lập:

- `kaggle-ml-experiment`
- `retail-forecast-evaluation`
- `thesis-experiment-report`
- `retail-code-verify`

## Cài trên Windows

1. Đóng các phiên Codex đang chạy công việc quan trọng.
2. Mở PowerShell tại thư mục project và chạy:

```powershell
Copy-Item -Recurse -Force .\custom-skills\kaggle-ml-experiment, .\custom-skills\retail-forecast-evaluation, .\custom-skills\thesis-experiment-report, .\custom-skills\retail-code-verify "$env:USERPROFILE\.codex\skills\"
```

3. Mở một lượt chat Codex mới. Skill chỉ được nạp ở lượt tiếp theo.

Để kiểm tra, yêu cầu rõ tên skill, ví dụ: `Dùng $retail-code-verify để kiểm tra và sửa lỗi trong code vừa tạo cho Notebook 03.`

Các skill không chứa token, thông tin đăng nhập, dữ liệu H&M hoặc mô hình. Bạn có thể giữ thư mục này trong project như bản nguồn để chỉnh sửa sau này.
