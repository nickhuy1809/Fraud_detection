# Credit Card Fraud Detection with Pure NumPy

Phát hiện giao dịch thẻ tín dụng gian lận sử dụng Machine Learning với NumPy thuần túy - không phụ thuộc vào thư viện ML có sẵn.

---

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Dataset](#dataset)
- [Cài đặt](#cài-đặt)
- [Cấu trúc Project](#cấu-trúc-project)
- [Kỹ thuật Implementation](#kỹ-thuật-implementation)
- [Kết quả](#kết-quả)
- [Tác giả](#tác-giả)

---

## Giới thiệu

### Mô tả bài toán

Gian lận thẻ tín dụng là một trong những vấn đề nghiêm trọng nhất trong ngành tài chính hiện đại. Mỗi năm, hàng tỷ đô la bị mất do các giao dịch gian lận, gây thiệt hại lớn cho cả ngân hàng và khách hàng. Bài toán đặt ra là: **Làm thế nào để tự động phát hiện các giao dịch gian lận trong thời gian thực từ hàng triệu giao dịch mỗi ngày?**

### Động lực và ứng dụng thực tế

1. **Bảo vệ tài chính**: Phát hiện sớm gian lận giúp ngăn chặn tổn thất tài chính cho khách hàng và tổ chức tài chính
2. **Giảm chi phí vận hành**: Tự động hóa việc phát hiện gian lận thay vì kiểm tra thủ công từng giao dịch
3. **Cải thiện trải nghiệm khách hàng**: Giảm thiểu tình trạng chặn nhầm giao dịch hợp lệ (false positive)
4. **Học tập sâu về ML**: Hiểu rõ bản chất hoạt động của các thuật toán thay vì chỉ gọi API

**Ứng dụng thực tế:**

- Hệ thống thanh toán trực tuyến (e-commerce, banking apps)
- Cảnh báo real-time khi phát hiện giao dịch khả nghi
- Phân tích forensics sau khi xảy ra gian lận
- Xây dựng hệ thống scoring rủi ro cho từng giao dịch

### Mục tiêu cụ thể

Project này tập trung vào việc 
- Thực hiện phân tích khám phá dữ liệu (EDA) nhằm hiểu rõ đặc điểm và mẫu hình của các giao dịch gian lận.  
- Phát hiện và phân tích các dấu hiệu bất thường, cũng như những insight đáng chú ý liên quan đến gian lận (nếu có).  
- Tiền xử lý và chuẩn hóa dữ liệu để chuẩn bị cho việc xây dựng các mô hình phát hiện gian lận.  
- xây dựng ba mô hình Machine Learning cổ điển để giải quyết bài toán phát hiện gian lận bằng cách chỉ sử dụng NumPy thuần túy:
1. **Linear Regression** 
2. **Logistic Regression** 
3. **Gaussian Naive Bayes**
---

##  Dataset

### Nguồn dữ liệu

Dataset được sử dụng là [**Credit Card Fraud Detection Dataset**](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download) 
**Chi tiết:**
- **File**: `creditcard.csv`
- **Đã được xử lý**: PCA transformation để bảo mật thông tin khách hàng
- **Kích thước**: Khoảng 150MB dữ liệu thô

### Mô tả các features

Dataset bao gồm **31 features** (30 input + 1 output):

#### Features đầu vào:

1. **Time** (1 feature)
   - Thời gian (giây) từ giao dịch đầu tiên trong dataset
   - Cho phép phân tích xu hướng theo thời gian
   - Range: 0 - 172,792 giây (khoảng 48 giờ)

2. **V1, V2, ..., V28** (28 features)
   - Kết quả của PCA transformation
   - Các features gốc đã được ẩn danh hóa để bảo mật
   - Mỗi feature V đại diện cho một principal component
   - Đã được chuẩn hóa (normalized)

3. **Amount** (1 feature)
   - Số tiền giao dịch (đơn vị: Euro)
   - Feature quan trọng cho fraud detection
   - Range: €0.00 - €25,691.16
   - Highly skewed distribution

#### Target variable:

4. **Class** (1 feature - output)
   - 0: Giao dịch bình thường (Normal)
   - 1: Giao dịch gian lận (Fraud)
   - Binary classification target

### Kích thước và đặc điểm dữ liệu

**Số liệu thống kê:**

```
Tổng số giao dịch:    284,807 transactions
├─ Normal (Class 0):  284,315 (99.83%)
└─ Fraud (Class 1):       492 ( 0.17%)

Imbalance Ratio:      577:1 (Normal:Fraud)
```

**Đặc điểm quan trọng:**

1. **Extreme Class Imbalance** ⚠️
   - Giao dịch gian lận chỉ chiếm 0.17% - cực kỳ hiếm
   - Random guessing accuracy: ~99.83% (vô nghĩa!)
   - Cần sử dụng class weights và metrics phù hợp (Precision, Recall, F1-Score)

2. **Dữ liệu đã được xử lý sẵn**
   - Features V1-V28 đã qua PCA transformation
   - Không cần feature engineering phức tạp
   - Tập trung vào modeling và xử lý imbalance

3. **Không có missing values**
   - Dataset hoàn chỉnh, không cần imputation
   - Tiết kiệm thời gian preprocessing

4. **Time-series component**
   - Có thể phân tích pattern theo thời gian
   - Fraud có xu hướng xảy ra vào giờ nào?
   - Có seasonality hay trend không?

5. **Amount distribution**
   - Normal transactions: Trung bình ~€88, median ~€22
   - Fraud transactions: Trung bình ~€122, median ~€9
   - Giao dịch gian lận có xu hướng giá trị thấp hơn

---

##  Phương pháp và Kỹ thuật Implementation

### Tổng quan về ba thuật toán

Project implement ba thuật toán Machine Learning cổ điển từ đầu, mỗi thuật toán có cách tiếp cận riêng để giải quyết bài toán phân loại:

1. **Linear Regression** - Mô hình hồi quy tuyến tính cơ bản
2. **Logistic Regression** - Mô hình phân loại nhị phân với sigmoid activation
3. **Gaussian Naive Bayes** - Mô hình xác suất dựa trên định lý Bayes

### 1. Linear Regression (Baseline Model)

#### Lý thuyết

Linear Regression dự đoán output liên tục bằng cách tìm mối quan hệ tuyến tính giữa input features và target:

**Hypothesis function:**
```
h(x) = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
     = wᵀx + b
```

**Cost function (Mean Squared Error với L2 Regularization):**
```
J(w) = (1/2m) Σᵢ₌₁ᵐ (h(xⁱ) - yⁱ)² + (λ/2) Σⱼ₌₁ⁿ wⱼ²
```
Trong đó:
- m: số lượng samples
- λ: regularization strength (tránh overfitting)
- w: weight vector
- b: bias term

**Gradient Descent update rules:**
```
∂J/∂wⱼ = (1/m) Σᵢ₌₁ᵐ (h(xⁱ) - yⁱ)xⱼⁱ + λwⱼ
∂J/∂b = (1/m) Σᵢ₌₁ᵐ (h(xⁱ) - yⁱ)

wⱼ := wⱼ - α(∂J/∂wⱼ)
b := b - α(∂J/∂b)
```

#### Implementation 

**Vectorized forward propagation:**
```python
# Shape: (m, n) @ (n,) + scalar = (m,)
y_pred = X @ weights + bias
```

**Efficient gradient computation sử dụng np.einsum:**
```python
# Thay vì: dw = (1/m) * X.T @ error
# Dùng einsum hiệu quả hơn:
dw = (1/m) * np.einsum('ij,i->j', X, error) + lambda_ * weights
```

**Xavier initialization** cho numerical stability:
```python
limit = np.sqrt(6.0 / (n_features + 1))
weights = np.random.uniform(-limit, limit, n_features)
```
---

### 2. Logistic Regression (Main Model)

#### Lý thuyết

Logistic Regression là model chuẩn cho binary classification, sử dụng sigmoid function để map output về xác suất [0, 1].

**Sigmoid activation function:**
```
σ(z) = 1 / (1 + e⁻ᶻ)
```

Properties của sigmoid:
- Output range: (0, 1) → diễn giải là xác suất
- Smooth gradient → dễ tối ưu hóa
- σ'(z) = σ(z)(1 - σ(z)) → đạo hàm đơn giản

**Hypothesis function:**
```
h(x) = σ(wᵀx + b) = 1 / (1 + e⁻⁽ʷᵀˣ ⁺ ᵇ⁾)
```

**Cost function (Binary Cross-Entropy với L2 Regularization):**
```
J(w) = -(1/m) Σᵢ₌₁ᵐ [yⁱ log(h(xⁱ)) + (1-yⁱ) log(1-h(xⁱ))] + (λ/2) Σⱼ wⱼ²
```

Cross-entropy loss phù hợp cho classification vì:
- Penalize heavily khi prediction sai (y=1 nhưng h(x)→0)
- Convex function → guaranteed convergence
- Probabilistic interpretation

**Gradient Descent update rules:**
```
∂J/∂wⱼ = (1/m) Σᵢ₌₁ᵐ (h(xⁱ) - yⁱ)xⱼⁱ + λwⱼ
∂J/∂b = (1/m) Σᵢ₌₁ᵐ (h(xⁱ) - yⁱ)

wⱼ := wⱼ - α(∂J/∂wⱼ)
b := b - α(∂J/∂b)
```

#### Xử lý Class Imbalance với Weighted Loss

Với tỷ lệ 577:1 (Normal:Fraud), model sẽ bias về class Normal. Giải pháp: **sample weighting**.

**Modified cost function:**
```
J(w) = -(1/m) Σᵢ₌₁ᵐ wᵢ[yⁱ log(h(xⁱ)) + (1-yⁱ) log(1-h(xⁱ))] + (λ/2) Σⱼ wⱼ²
```

Trong đó:
- wᵢ = class_weight[yⁱ]
- class_weight[0] = 1.0 (Normal)
- class_weight[1] = 577.0 (Fraud) → tăng importance của fraud samples

#### Numerical Stability Techniques

**1. Clipping để tránh overflow:**
```python
z = np.clip(z, -500, 500)  # Prevent exp(-500) = 0
sigmoid = 1.0 / (1.0 + np.exp(-z))
```

**2. Epsilon cho log stability:**
```python
epsilon = 1e-15
y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
# Tránh log(0) = -∞
```

#### Implementation 

**Vectorized sigmoid:**
```python
def _sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))
```

**Sample weight assignment sử dụng fancy indexing:**
```python
sample_weights = np.zeros(len(y))
for class_label, weight in class_weights.items():
    mask = (y == class_label)  # Boolean mask
    sample_weights[mask] = weight  # Vectorized assignment
```

**Weighted gradient với einsum:**
```python
weighted_error = sample_weights * error
dw = (1/m) * np.einsum('ij,i->j', X, weighted_error) + lambda_ * weights
```

---

### 3. Gaussian Naive Bayes (Probabilistic Model)

#### Lý thuyết

Naive Bayes dựa trên **Bayes' Theorem** với assumption rằng các features độc lập với nhau (naive assumption).

**Bayes' Theorem:**
```
P(y|x) = P(x|y)P(y) / P(x)
```

Với classification:
```
ŷ = argmax_y P(y|x) = argmax_y [P(x|y)P(y)]
```

**Gaussian Naive Bayes assumption:**

Mỗi feature xᵢ tuân theo phân phối Gaussian với mean μᵢ và variance σᵢ² khác nhau cho mỗi class:

```
P(xᵢ|y=c) = (1/√(2πσᵢ,c²)) exp(-(xᵢ - μᵢ,c)²/(2σᵢ,c²))
```

**Independence assumption (Naive):**
```
P(x|y=c) = ∏ᵢ₌₁ⁿ P(xᵢ|y=c)
```

**Final prediction:**
```
P(y=c|x) ∝ P(y=c) ∏ᵢ₌₁ⁿ P(xᵢ|y=c)
```

#### Log-space computation để tránh underflow

Nhân nhiều xác suất nhỏ → underflow (→ 0). Giải pháp: **work in log-space**.

**Log-likelihood:**
```
log P(y=c|x) = log P(y=c) + Σᵢ log P(xᵢ|y=c)
             = log πc + Σᵢ [-1/2 log(2πσᵢ,c²) - (xᵢ-μᵢ,c)²/(2σᵢ,c²)]
```

**Mahalanobis distance** (efficient với einsum):
```
d² = Σᵢ (xᵢ - μᵢ)² / σᵢ²
```

**Log-sum-exp trick** cho normalization:
```
log P(y=c|x) = log_likelihood_c - log(Σc' exp(log_likelihood_c'))
```

NumPy provides: `np.logaddexp.reduce()` để compute log-sum-exp stably.

#### Training process

**1. Compute class priors:**
```
P(y=c) = (# samples in class c) / (total # samples)
```

**2. Compute class-conditional statistics:**

Cho mỗi class c:
```
μᵢ,c = mean(xᵢ | y=c)
σᵢ,c² = var(xᵢ | y=c) + ε
```

ε (var_smoothing) prevents division by zero.

**3. Prediction:** Compute log-likelihood cho mỗi class, chọn class có likelihood cao nhất.

#### Implementation

**Vectorized statistics computation:**
```python
# Class-wise mean và variance
for idx, class_label in enumerate(classes):
    class_mask = (y == class_label)  # Boolean mask
    X_class = X[class_mask]          # Fancy indexing
    
    # Broadcasting: compute mean/var across axis 0
    means[idx] = np.mean(X_class, axis=0)
    vars[idx] = np.var(X_class, axis=0) + var_smoothing
```

**Efficient Mahalanobis distance sử dụng einsum:**
```python
diff = X - means[class_idx]  # Broadcasting: (n_samples, n_features)
# Compute: Σᵢ (xᵢ - μᵢ)² / σᵢ²
mahal_dist = np.einsum('ij,j,ij->i', diff, 1.0/vars[class_idx], diff)
```

**Log-sum-exp trick:**
```python
# Normalize log probabilities
log_sum_exp = np.logaddexp.reduce(log_posterior, axis=1, keepdims=True)
log_proba = log_posterior - log_sum_exp
```

---
### So sánh ba thuật toán

| Aspect | Linear Regression | Logistic Regression | Naive Bayes |
|--------|------------------|---------------------|-------------|
| **Output** | Continuous | Probability [0,1] | Probability [0,1] |
| **Decision boundary** | Linear | Linear (sigmoid) | Non-linear (Gaussian) |
| **Loss function** | MSE | Cross-Entropy | Likelihood |
| **Optimization** | Gradient Descent | Gradient Descent | Closed-form (statistics) |
| **Assumption** | Linear relationship | Linear separation | Feature independence |
| **Training speed** | Medium | Medium | Fast (no iteration) |
| **Handle imbalance** | Poor | Good (với weights) | Good (natural) |
| **Interpretability** | High | High | Medium |
| **Overfitting risk** | Medium | Medium (với L2) | Low |

**Khi nào dùng model nào?**

- **Linear Regression**: Baseline, regression tasks
- **Logistic Regression**: Binary classification, cần interpretability, linear separable data
- **Naive Bayes**: Fast prediction, feature independence holds, small training data

---

## Cài đặt

### Các bước cài đặt

**1. Clone repository**

```bash
git clone https://github.com/nickhuy1809/Fraud_dectection.git
cd Fraud_dectection
```
**2. Cài đặt dependencies**

```bash
pip install -r requirements.txt
```

## Cấu trúc Project

```
Fraud_dectection/
│
├── data/ 
│   └── creditcard.csv            # Dataset chính
│
├── notebooks/                      
│   ├── 01_data_exploration.ipynb       # Khám phá và phân tích dữ liệu
│   ├── 02_preprocessing.ipynb          # Tiền xử lý và feature engineering
│   └── 03_modeling.ipynb               # Training và evaluation 3 models
│
├── requirements.txt                    # Dependencies
├── README.md                           # Tài liệu dự án
```

---

## Challenges & Solutions

### Thách thức khi implement với NumPy thuần túy

#### 1. **Numerical Instability**

**Vấn đề:**
- Sigmoid overflow/underflow khi `exp()` với giá trị quá lớn/nhỏ
- Log của xác suất rất nhỏ gây ra -∞
- Nhân nhiều xác suất nhỏ trong Naive Bayes dẫn đến underflow

**Giải pháp:**
- Clipping values trước khi tính sigmoid: `z = np.clip(z, -500, 500)`
- Thêm epsilon để tránh log(0): `y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)`
- Log-sum-exp trick cho Naive Bayes: `np.logaddexp.reduce()`
- Work in log-space thay vì probability space

---

#### 2. **Vectorization Complexity**

**Vấn đề:**
- Một số operations khó vectorize hoàn toàn (stratified split, class-wise statistics)
- Code phức tạp, khó debug hơn Python loops
- Shape mismatches khó phát hiện

**Giải pháp:**
- Hybrid approach: Loop over classes (2 iterations) + vectorize operations trong mỗi class
- Boolean masking và fancy indexing: `X[y == class_label]`
- Print shapes và assertions để debug
- Test với toy data nhỏ trước khi chạy full dataset
---

#### 3. **Không có built-in features**

**Vấn đề:**
- Phải implement từ đầu: StandardScaler, train_test_split, metrics
- Dễ có bugs trong implementation
- Thiếu các utilities tiện lợi của scikit-learn

**Giải pháp:**
- Modular design: Tách thành các functions/classes nhỏ, dễ test
- Unit tests để verify implementation
- Extensive input validation
- Code reuse: Tạo utility functions cho operations dùng nhiều lần
---

## Future Improvements

### Hướng phát triển tiếp theo

#### 1. **Feature Engineering**

- **Time features**: Extract hour_of_day, day_of_week từ Time column
- **Transaction velocity**: Số lượng giao dịch trong khung thời gian gần nhất
- **Amount features**: Log transformation, binning theo ranges

#### 2. **Model Optimization & Validation**

- **Hyperparameter tuning**: Grid search cho learning_rate, lambda_, n_iterations
- **Cross-validation**: K-fold stratified CV để estimate performance reliably
- **Production deployment**: Save/load models, REST API cho real-time prediction
---

## Tác giả

Họ và tên: **Cao Tấn Hoàng Huy**  
MSSV: **23127051**
Email: **cthhuy23@clc.fitus.edu.vn**
Trường Đại học Khoa học tự nhiên, ĐHQG-HCM (HCMUS)
---
##  License

MIT License

Copyright (c) 2025 nickhuy1809
---