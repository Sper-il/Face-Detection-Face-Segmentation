# HƯỚNG DẪN TẠO template.docx CHO PANDOC

File template.docx là file Word mẫu được Pandoc dùng để định dạng khi chuyển đổi từ Markdown sang Word (.docx). Các styles trong template sẽ được áp dụng cho toàn bộ báo cáo.

## Các bước tạo template.docx

### Cách 1: Tạo thủ công trong Word

1. **Mở Microsoft Word** và tạo file mới.

2. **Đặt font mặc định:**
   - Vào `Home` -> `Font` -> chọn `Times New Roman` size `13`.
   - Bôi đen toàn bộ văn bản (`Ctrl + A`) -> nhấn `Ctrl + D` để mở Font dialog -> chọn `Times New Roman`, size `13`, color `Auto`.

3. **Căn lề trang:**
   - `Layout` -> `Margins` -> `Custom Margins`
   - Top: 2.5 cm
   - Bottom: 2.5 cm
   - Left: 3 cm
   - Right: 2 cm
   - Orientation: Portrait

4. **Đặt dãn dòng:**
   - `Home` -> `Paragraph` -> `Line Spacing` -> `1.5`

5. **Tạo các styles (cần thiết cho Pandoc):**

   Mở **Styles pane** (`Alt + Ctrl + Shift + S` hoặc Home -> Styles), sau đó tạo/sửa các styles sau:

   | Style name      | Font           | Size | Bold | Italic | Alignment  | Spacing |
   |-----------------|----------------|------|------|--------|------------|---------|
   | Normal          | Times New Roman| 13pt | No   | No     | Justify    | 1.5     |
   | Heading 1       | Times New Roman| 16pt | Yes  | No     | Center     | 1.5     |
   | Heading 2       | Times New Roman| 14pt | Yes  | No     | Center     | 1.5     |
   | Heading 3       | Times New Roman| 13pt | Yes  | No     | Left       | 1.5     |
   | Heading 4       | Times New Roman| 13pt | Yes  | Yes    | Left       | 1.5     |
   | Title           | Times New Roman| 18pt | Yes  | No     | Center     | 1.5     |
   | Subtitle        | Times New Roman| 14pt | No   | Yes    | Center     | 1.5     |
   | Author          | Times New Roman| 13pt | Yes  | No     | Center     | 1.5     |
   | Date            | Times New Roman| 13pt | No   | No     | Center     | 1.5     |
   | Caption         | Times New Roman| 13pt | No   | Yes    | Center     | 1.0     |
   | Strong Emphasis | Times New Roman| 13pt | Yes  | No     | Justify    | 1.5     |
   | Emphasis        | Times New Roman| 13pt | No   | Yes    | Justify    | 1.5     |
   | Code Block      | Consolas       | 11pt | No   | No     | Left       | 1.0     |
   | Source Code     | Consolas       | 11pt | No   | No     | Left       | 1.0     |
   | TOC 1           | Times New Roman| 13pt | Yes  | No     | Left       | 1.0     |
   | TOC 2           | Times New Roman| 13pt | No   | No     | Left       | 1.0     |
   | TOC 3           | Times New Roman| 12pt | No   | No     | Left       | 1.0     |
   | Block Text      | Times New Roman| 13pt | No   | No     | Justify    | 1.5     |

6. **Cách sửa styles trong Word:**
   - Trong Styles pane, nhấp chuột phải vào style (ví dụ "Heading 1") -> "Modify"
   - Đặt Font, Size, Bold, Italic, Alignment theo bảng trên
   - Nhấn OK

7. **Lưu file** với tên `template.docx` ở cùng thư mục với `BAO_CAO_CHI_TIET.md`.

### Cách 2: Tạo tự động bằng pandoc

Nếu đã cài pandoc, có thể tạo template mặc định từ file rỗng:

```bash
pandoc -o template.docx --print-default-data-file reference.docx
```

File này sẽ là template mặc định. Sau đó chỉnh sửa styles trong Word theo bảng trên.

## Chuyển đổi Markdown -> Word với template

```bash
pandoc BAO_CAO_CHI_TIET.md -o BAO_CAO.docx \
  --reference-doc=template.docx \
  --toc \
  --toc-depth=3 \
  --number-sections
```

## Chuyển đổi Markdown -> PDF

Nếu muốn xuất PDF trực tiếp (cần cài LaTeX):

```bash
pandoc BAO_CAO_CHI_TIET.md -o BAO_CAO.pdf \
  --pdf-engine=xelatex \
  --variable mainfont="Times New Roman" \
  --variable geometry="a4paper, margin=2.5cm, left=3cm" \
  --variable fontsize=13pt \
  --toc \
  --toc-depth=3 \
  --number-sections
```

Hoặc dùng wkhtmltopdf:

```bash
pandoc BAO_CAO_CHI_TIET.md -o BAO_CAO.html
wkhtmltopdf BAO_CAO.html BAO_CAO.pdf
```

## Lưu ý về font Times New Roman

Để font Times New Roman hiển thị đúng:

1. **Trên Windows:** Font có sẵn theo mặc định.
2. **Trên Linux/Mac:** Có thể cần cài đặt:
   ```bash
   # Ubuntu
   sudo apt install ttf-mscorefonts-installer
   # macOS
   brew install --cask font-times-new-roman
   ```

## Thông tin thêm về Pandoc

- Trang chủ: https://pandoc.org/
- Hướng dẫn chi tiết: https://pandoc.org/MANUAL.html
- Template mẫu: https://github.com/jgm/pandoc-templates

## Script tự động tạo template

Lưu file `make_template.ps1` (PowerShell):

```powershell
# Generate a default Pandoc Word template
pandoc -o template.docx --print-default-data-file reference.docx
Write-Host "Template created: template.docx"
Write-Host "Now open it in Word and modify styles as described above."
```

Hoặc `make_template.bat`:

```batch
@echo off
pandoc -o template.docx --print-default-data-file reference.docx
echo Template created: template.docx
echo Now open it in Word and modify styles.
pause
```

## Kiểm tra template hoạt động đúng

Sau khi tạo template, chạy lệnh sau để kiểm tra:

```bash
pandoc BAO_CAO_CHI_TIET.md -o test.docx --reference-doc=template.docx
```

Mở `test.docx` trong Word và kiểm tra:
- [ ] Font chữ là Times New Roman
- [ ] Heading 1 size 16pt, in đậm, canh giữa
- [ ] Heading 2 size 14pt, in đậm, canh giữa
- [ ] Heading 3 size 13pt, in đậm, canh trái
- [ ] Body text size 13pt, dãn dòng 1.5
- [ ] Code block dùng font Consolas/Courier
- [ ] Lề trái 3cm, lề phải 2cm
