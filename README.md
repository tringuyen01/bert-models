# bert-models
Dataset: https://github.com/duyvuleo/VNTC
Các bước xử lý data:
  - Đọc data từ file train và test từ hàm read_file(category, file_name), lưu vào mảng với mỗi phần tử là một cặp dữ liệu và loại văn bản 
  - Chia data thành tập train và validation từ hàm train_test_split(datanews, test_size = 0.1).
  - Convert vào thư viện pandas.
  - Encoder label thành kiểu int.
  - Tách các từ thành các input_ids, thêm token [cls] vào đầu và token [sep] vào cuối (đã được mã hóa thành input_ids), cắt dữ liệu đưa vào dựa vào max length 
  và khởi tạo attention mask sau đó đưa vào model để thực hiện training. [tokenizer.encode_plus]
