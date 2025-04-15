# DNGR-Keras
Deep Neural Networks for Learning Graph Representations (DNGR)
This repository implements [DNGR](https://github.com/ShelsonCao/DNGR.git) using keras,example.py demonstrates its application on wine dataset. 

# Giải thích file input của graph
Cần có:
1. `wine.edgelist`: file tập cạnh cùng trọng số (thông thường ta có thể set up trọng số = 1)
2. `wine_network.mat`: file ma trận kề/trọng số (thật ra file này cũng có thể suy ra từ file 1 nhưng đòi hỏi thêm code 1 chút)
3. `wine_label.mat`: label của các communities

# Hướng dẫn chạy code 
1. Khởi tạo 1 môi trường conda mới, như đã hướng dẫn tại repo [VGAER](https://github.com/Dat-DangTien/VGAER#) (giả sử môi trường tên là `dngr_env`)
2. Mở môi trường vừa tạo và cài đủ package như file `requirements.txt`
```bash
pip install -r requirements.txt
```
3. Chuyển terminal vào thư mục hiện tại là `DNGR-Keras`, lúc đó terminal cần hiện:
```bash
(dngr_env) <your_local_path>\DNGR-Keras> 
```
3. Gen ra file `represenation.pkl` - file biểu diễn:
```bash
python DNGR.py --graph_type undirected --input wine.edgelist --output representation
```
4. Chạy thử file `example.py`:
```bash
python example.py
```
