# Mô tả chương trình
Chương trình được dùng để đánh giá một sinh viên năm 2 hoặc năm 3 có phù hợp với ngành học công nghệ thông tin hay không , chương trình được xây dựng trên thuật toán Random Forest để dự đoán thu nhập của sinh viên sau khi ra trường , sau đó đánh giá sinh viên dựa trên mức thu nhập đó               (nếu >= 6 000 000 là phù hợp , < 6 000 000 là chưa phù hợp ). Tập dự liệu được dùng để huấn luyện mô hình được lấy từ cơ sở dữ liệu của trung tâm quản lí chất lượng - Đại học công nghiệp Hà Nội. Tập dữ liệu huấn luyên bao gồm các bản ghi chứa các thông tin về điểm các học phần  , điểm trung bình trung tích  lũy , quê quán , giới tính , thu nhập của sinh viên sau khi ra trường. Do dữ liệu thu thập được từ các sinh viên đã đi làm còn quá ít, chỉ bao gồm 214 bản ghi, vậy nên chương trình sẽ tự động dự đoán thu nhập của các sinh viên đã đi làm nhưng không thu thập được thông tin về lương, sau đó sẽ lại huấn luyện một lần nữa dựa trên các thông tin đã được sinh tự động và thông tin thu thập được từ thực tế. Việc dự đoán thu nhập sau khi ra trường là một dạng bài toán Regression trong Machine Learning với Input đầu vào là các features điểm các học phần , điểm trung bình trung tích lũy và Ouput là thu nhập của người đó. Mô hình Random Forest được lựa chọn vì cho kết quả tốt .


# Chuẩn bị dữ liệu 

### **Tập dữ liệu được sử dụng cho mô hình thứ 1 (dự đoán thu nhập của những sinh viên không có dự liệu thu nhập)**

### *Tập dữ liệu huấn luyện*
X_train : 214 bản ghi , 15 thuộc tính (gồm điểm các môn học , điểm tích lũy , quê quán , giới tính)
Y_train : 214 bản ghi , 1 thuộc tính (thu nhập sau khi ra trường)


### *Tập dữ liệu kiểm thử*
X_test : 2071 bản ghi , 15 thuộc tính (gồm điểm các môn học , điểm tích lũy , quê quán , giới tính) 
salary : 2071 bản ghi , 1 thuộc tính (thu nhập của sinh viên được dự đoán bằng mô hình đã huấn luyện )

### **Tập dữ liệu được dùng cho cho mô hình thứ 2 (mô hình dự đoán thu nhập của sinh viên sau khi ra trường)**

### *Tập dữ liệu huấn luyện*
X_train : 1396 bản ghi , 15 thuộc tính (gồm điểm các môn học , điểm tích lũy , quê quán , giới tính)
Y_train : 1396 bản ghi , 1 thuộc tính (thu nhập sau khi ra trường)


### *Tập dữ liệu kiểm thử*
X_test : 689 bản ghi , 15 thuộc tính (gồm điểm các môn học , điểm tích lũy , quê quán , giới tính) 
salary : 689 bản ghi , 1 thuộc tính (thu nhập của sinh viên được dự đoán bằng mô hình đã huấn luyện )



# Features
Các đặc trưng được sử dụng để đưa vào mô hình dự đoán bao gồm các môn học quan trọng trong ngành công nghệ thông tin :
- Toán rời rạc
- Cơ sở dữ liệu
- Kỹ thuật lập trình
- Mạng máy tính
- Cấu trúc dữ liệu và giải thuật
- Lập trình hướng đối tượng
- Phân tích thiết kế hệ thống
- Thiết kế web
- Công nghệ XML
- Kỹ thuật lập trình
- Điểm trung bình trung tích lũy
- Quê quán
- Giới tính

# Huấn luyện mô hình

### Mô hình thứ 1
```
from sklearn.ensemble import RandomForestRegressor

Fit model
random_forest_model = RandomForestRegressor(n_jobs = -1, min_samples_leaf = 3 , n_estimators = 200)
random_forest_model.fit(X_train , Y_train)
random_forest_model.score(X_train , Y_train)

Predict
predict = random_forest_model.predict(X_test) 
salary = pd.DataFrame(predict, columns = ['salary'])

```

### Mô hình thứ 2

```
Fit model
random_forest_model = RandomForestRegressor(n_jobs = -1, min_samples_leaf = 3 , n_estimators = 200)
random_forest_model.fit(x_train , y_train)
random_forest_model.score(x_train , y_train)

Predict
predict = random_forest_model.predict(x_test)
predict_df = pd.DataFrame({'salary' : predict})

```


# Đánh giá mô hình
Sử dụng r2 metric để đánh giá mô hình , với phương pháp r2_metric thì kết quả tốt nhất sẽ là 1.0

```
from sklearn.metrics import r2_score

score  = r2_score(y_test , predict)
print(score)

0.09018553808926699
```

# Đánh giá sinh viên dựa trên mức lương và in thông tin sinh viên ra file cvs
```
rdf_predict_df['danhgia'] = rdf_predict_df['salary'].apply(lambda x : 'không phù hợp' if x < 6000000 else 'phù hợp')
#In thông tin ra file csv
rdf_predict_df.to_csv('./danhgiasinhvien.csv' , index = False)
```
