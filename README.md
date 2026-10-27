# Công cụ chữ tay của hồ phá thơ

Nhánh này chỉ để cất công cụ. Trang hophatho.ink chạy từ nhánh `main` và không dùng gì ở đây.

## Có gì

- `font/`: ảnh các trang vở ô ly Nguyên viết tay
  - `src2/`: chữ thường, số, dấu câu, dấu thanh, âm ghép, dạng cuối từ (7 trang)
  - `src2b/`: trang bổ sung
  - `src2c/`: chữ hoa (2 trang; chữ gạch bằng mực cam là bỏ)
  - `src2d/`: chữ in hoa, dùng cho chữ viết tắt như LGBTQIA
  - `src2e/`: Ơ, Ư hoa
- `font3/`
  - `db3.py`: tách từng chữ ra khỏi ảnh, đặt chữ lên dòng kẻ
  - `clean.py`: xoá dòng kẻ ô ly, giữ nét mực
  - `build_font3.py`: dựng font từ các chữ đã tách
  - `preview4.py`, `sheet.py`: in thử font ra ảnh
- `haiku/ink.py`: tách nét chữ viết tay của hai bài haiku (mướp chín, lông tay) thành ảnh nền trong suốt, kèm hai trang gốc
- `tex/tex_update.py`: tính màu trung bình, chỗ sáng, chỗ tối và màu chữ gợi ý cho ảnh nền chất liệu của một bài thơ

## Dựng lại font

```
pip install -r requirements.txt
cd font3
python3 db3.py
python3 build_font3.py work/HPTTay.ttf
python3 -c "from fontTools.ttLib import TTFont; f = TTFont('work/HPTTay.ttf'); f.flavor = 'woff2'; f.save('work/hpt-tay.woff2')"
```

Sau đó thay `chu/hpt-tay.woff2` trên nhánh `main` bằng `font3/work/hpt-tay.woff2`, rồi đổi số `?v=` ở hai chỗ gọi font trong `index.html` (dòng `preload` và `@font-face`) thành 8 ký tự đầu của mã md5 của tệp font mới. Không đổi số này thì Safari có thể vẫn giữ font cũ.

Dựng lại từ đây ra đúng font đang chạy trên trang (28/10/2026).

## Thêm chữ hoặc viết lại chữ

Chụp hoặc scan trang vở mới, để vào `font/`, rồi thêm một dòng tách trang đó trong `db3.py`. Trang Ơ, Ư hoa (`caps_page('nh3', ...)`) là ví dụ ngắn nhất. Trang nào mực nhạt thì tăng `hi` như trang `nh2`.
