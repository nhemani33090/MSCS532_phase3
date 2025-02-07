# 🚀 Optimized Recommendation System  

## 📖 Overview  
A fast and scalable **recommendation system** using **Annoy (Approximate Nearest Neighbors), Sparse Matrices, and SQLite**.  
It efficiently stores **user-product interactions** and performs **similarity searches** with **multi-threading** for faster recommendations.

---

## 🛠 Requirements  
Make sure you have **Python 3.x** installed. Then, install the required libraries:  
```sh
pip3 install numpy scipy sqlite3 annoy matplotlib psutil
```

---

## ▶️ How to Run  
```sh
python3 data_structures.py
```
- This will **create the database**, generate **product features**, and **compute recommendations**.
- **Automatically runs tests**, including:
  - **Similarity searches**
  - **Performance benchmarking (Annoy vs. KD-Tree)**
  - **Memory usage analysis**

---

## 📌 Notes  
- **Database (`recommendation.db`) is auto-generated.** No need to create it manually.  
- **Multi-threading is used for faster similarity searches.**  
- **Uses Sparse Matrices** to optimize memory usage.  

---
