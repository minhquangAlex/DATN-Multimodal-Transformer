# Kết nối VS Code, Kaggle và Overleaf

## Mô hình đề xuất

GitHub là nơi chia sẻ **mã nguồn và notebook**. Dữ liệu H&M gốc, dữ liệu đã xử lý, embedding, kết quả train và file zip được giữ ngoài GitHub; `.gitignore` đã loại các thư mục này. Kaggle tiếp tục lưu notebook và chạy GPU. Overleaf chứa phần báo cáo LaTeX riêng, không chứa toàn bộ source và dữ liệu của project.

```text
VS Code / D:\DATN  <---->  GitHub repository
                                  |            \
                        Kaggle CLI             Overleaf Git
                   pull/push notebook        đồng bộ LaTeX
```

Kaggle CLI đẩy một phiên bản notebook lên Kaggle; đây không phải đồng bộ trực tiếp, liên tục. Trước khi đẩy một phiên bản, kiểm tra `kernel-metadata.json`, file notebook và data source. `kaggle kernels push` có thể chạy notebook; dùng `--no-run` nếu chỉ cần lưu phiên bản mã. Có thể dùng `kaggle kernels pull` để lấy mã notebook hiện tại về local.

## Cấp quyền cho Codex

1. Chủ repo GitHub mời **tài khoản GitHub của bạn** vào repo với quyền phù hợp, thường là Write nếu cần sửa/push.
2. Kết nối GitHub trong Codex bằng tài khoản của bạn và chỉ cấp quyền với repository này nếu giao diện hỗ trợ chọn repo.
3. Sau khi kết nối được xác nhận, Codex có thể làm việc với nội dung GitHub trong phạm vi quyền đã cấp.

Không gửi mật khẩu, Kaggle API token, Overleaf Git token hoặc file `kaggle.json` trong chat hay commit chúng vào repo. Nếu dùng Kaggle CLI, đăng nhập token trên máy của bạn theo hướng dẫn Kaggle; credentials phải nằm ngoài project.

## Nối notebook Kaggle hiện tại

1. Lấy owner/slug từ URL notebook Kaggle, dạng `owner/notebook-slug`.
2. Tải notebook và metadata về một thư mục làm việc riêng bằng `kaggle kernels pull owner/notebook-slug -p <folder> -m`.
3. Đưa notebook cần quản lý vào GitHub repo. Giữ notebook mirror trong thư mục riêng để tránh ghi đè bản hiện tại trong `notebooks/` trước khi đối chiếu thay đổi.
4. Khi cần phát hành thay đổi, kiểm tra metadata và dependency/data source, sau đó chạy `kaggle kernels push -p <folder>`. Lệnh này sẽ tải mã lên và thường bắt đầu chạy; dùng `--no-run` nếu chỉ muốn lưu phiên bản.
5. Tải output về bằng `kaggle kernels output owner/notebook-slug -p <output-folder>` khi cần lấy artifact.

Lần đầu cần xác nhận notebook thuộc tài khoản có quyền push. Kaggle kernel cần được tạo/lưu phiên bản trên giao diện trước khi cập nhật bằng CLI trong một số trường hợp.

## Nối báo cáo Overleaf

1. Chủ project Overleaf mời email tài khoản Overleaf của bạn với vai trò **Editor**. Khi nhiều collaborator cần quyền edit, khả năng này phụ thuộc gói Overleaf hiện hành.
2. Chọn một thư mục/repository chỉ dành cho luận văn, ví dụ `thesis/` hoặc repo riêng. Chỉ đưa `.tex`, `.bib`, hình và bảng cần cho báo cáo vào đó.
3. Overleaf Git integration cho phép collaborator có quyền truy cập clone/pull/push khi tính năng/gói phù hợp. Người dùng tạo Git authentication token riêng trong Overleaf; không chia sẻ token.
4. Nếu sử dụng GitHub Sync, nó là tính năng premium và Overleaf yêu cầu tạo project từ repo hiện có hoặc repo từ project hiện có; không thể ghép trực tiếp một Overleaf project đã tồn tại với một GitHub repo đã tồn tại qua GitHub Sync. Với project/repo đã tồn tại, dùng Git integration như một remote hoặc chọn repo/project mới.
5. Đồng bộ thủ công và kéo thay đổi trước khi đẩy để giảm xung đột. Overleaf chỉ hỗ trợ một nhánh tuyến tính cho Git integration; không dùng Git LFS/submodule trong phần Overleaf.

## Trạng thái workspace hiện tại

- Git local dùng nhánh `main`, remote `origin`: `https://github.com/minhquangAlex/DATN-Multimodal-Transformer.git`.
- Kaggle notebook `alexnguyen147/datn-notebook` đã được tải cùng metadata vào `integrations/kaggle/datn-notebook/`. Dùng `scripts/sync_kaggle.ps1` để pull, save không chạy, xem trạng thái hoặc tải output.
- Chưa có file LaTeX (`.tex`/`.bib`) trong project.
- GitHub có thể đồng bộ qua Git local. Overleaf chưa kết nối vì chưa có URL project/quyền truy cập; không có connector Overleaf trực tiếp trong workspace hiện tại.

## Thông tin cần để hoàn tất liên kết

- URL repo GitHub sẽ dùng, hoặc xác nhận cần tạo repo mới.
- URL Kaggle notebook hiện tại và quyền sở hữu/quyền sửa của tài khoản Kaggle.
- URL project Overleaf và xác nhận tài khoản của bạn đã được mời làm Editor.

Chỉ cần cung cấp URL/slug và trạng thái quyền. Không gửi token hoặc mật khẩu.
