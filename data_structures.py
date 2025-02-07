import sqlite3
import numpy as np
import time
import threading
import matplotlib.pyplot as plt
import psutil
from scipy.sparse import csr_matrix
from annoy import AnnoyIndex
from scipy.spatial import KDTree
import random

# ===========================
#  PHASE 3: OPTIMIZED SYSTEM
# ===========================

# Database Initialization
class UserDatabase:
    def __init__(self, db_name="recommendation.db"):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                user_id INTEGER,
                product_id INTEGER,
                rating REAL,
                PRIMARY KEY (user_id, product_id)
            )
        """)
        self.conn.commit()

    def add_user(self, user_id, name):
        self.cursor.execute("INSERT OR IGNORE INTO users (id, name) VALUES (?, ?)", (user_id, name))
        self.conn.commit()

    def add_interactions_batch(self, interactions):
        self.cursor.executemany("INSERT OR REPLACE INTO interactions (user_id, product_id, rating) VALUES (?, ?, ?)", interactions)
        self.conn.commit()

# Sparse Matrix for User-Product Interactions
class InteractionStore:
    def __init__(self):
        self.data = []
        self.rows = []
        self.cols = []

    def add_interaction(self, user_id, product_id, rating):
        self.rows.append(user_id)
        self.cols.append(product_id)
        self.data.append(rating)

    def get_sparse_matrix(self):
        return csr_matrix((self.data, (self.rows, self.cols)))

# Annoy-Based Product Similarity Matching
class ProductMatcher:
    def __init__(self, feature_size=100):
        self.feature_size = feature_size
        self.annoy_index = AnnoyIndex(self.feature_size, 'angular')

    def add_product(self, product_id, features):
        self.annoy_index.add_item(product_id, features)

    def build_index(self, num_trees=10):
        self.annoy_index.build(num_trees)

    def find_similar(self, product_id, n=5):
        return self.annoy_index.get_nns_by_item(product_id, n)

# Multi-threaded Similarity Search
def threaded_similarity_search(matcher, product_id, result_container):
    result_container[product_id] = matcher.find_similar(product_id)

def batch_similarity_search(matcher, product_ids):
    threads = []
    results = {}

    for product_id in product_ids:
        thread = threading.Thread(target=threaded_similarity_search, args=(matcher, product_id, results))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return results

# ===========================
#  ADDING MORE USER INTERACTIONS
# ===========================

db = UserDatabase()

# Simulate 1000 Users
num_users = 1000
num_products = 500
users = [(i, f"User_{i}") for i in range(1, num_users + 1)]

# Add users to the database
for user_id, name in users:
    db.add_user(user_id, name)

# Generate Random Interactions (Each user rates 10 random products)
interactions = []
for user_id in range(1, num_users + 1):
    rated_products = random.sample(range(100, 100 + num_products), 10)  # Each user rates 10 products
    for product_id in rated_products:
        rating = round(random.uniform(1.0, 5.0), 1)  # Random rating between 1.0 and 5.0
        interactions.append((user_id, product_id, rating))

# Batch Insert into Database
db.add_interactions_batch(interactions)

# ===========================
#  TESTING WITH SPARSE MATRIX
# ===========================

interaction_store = InteractionStore()
for user_id, product_id, rating in interactions:
    interaction_store.add_interaction(user_id, product_id, rating)

sparse_matrix = interaction_store.get_sparse_matrix()
print(f"Sparse Matrix Shape: {sparse_matrix.shape}")

# ===========================
#  SIMILARITY SEARCH TESTING
# ===========================

matcher = ProductMatcher(feature_size=5)

# Add 500 products with random feature vectors
for product_id in range(100, 100 + num_products):
    matcher.add_product(product_id, np.random.rand(5))

matcher.build_index()

# Find Similar Products for a Sample Set
test_product_ids = random.sample(range(100, 100 + num_products), 5)
similar_products = batch_similarity_search(matcher, test_product_ids)
print("Similar Products:", similar_products)

# ===========================
#  PERFORMANCE TESTING: KD-TREE vs. ANNOY
# ===========================

# Generate Random Product Features
num_products = 10000
feature_size = 10
product_features = np.random.rand(num_products, feature_size)

# KD-Tree (Phase 2)
kd_tree = KDTree(product_features)

# Annoy (Phase 3)
annoy_index = AnnoyIndex(feature_size, 'angular')
for i in range(num_products):
    annoy_index.add_item(i, product_features[i])
annoy_index.build(10)

# Query for a Random Product
query = np.random.rand(1, feature_size)

# Benchmark KD-Tree (Phase 2)
start_time = time.time()
kd_tree.query(query, k=5)
kd_time = time.time() - start_time

# Benchmark Annoy (Phase 3)
start_time = time.time()
annoy_index.get_nns_by_vector(query[0], 5)
annoy_time = time.time() - start_time

print(f"KD-Tree Time (Phase 2): {kd_time:.6f} seconds")
print(f"Annoy Time (Phase 3): {annoy_time:.6f} seconds")

# ===========================
#  MEMORY USAGE ANALYSIS
# ===========================
num_users, num_products = 10000, 5000
dense_matrix = np.random.rand(num_users, num_products)

rows, cols = np.nonzero(dense_matrix > 0.5)
data = dense_matrix[rows, cols]
sparse_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_products))

print(f"Dense Matrix Memory Usage: {dense_matrix.nbytes / 1e6:.2f} MB")
print(f"Sparse Matrix Memory Usage: {sparse_matrix.data.nbytes / 1e6:.2f} MB")

# ===========================
#  SCALABILITY TEST
# ===========================

dataset_sizes = [1000, 5000, 10000, 50000, 100000, 500000]
query_times = []

for size in dataset_sizes:
    annoy_index = AnnoyIndex(5, 'angular')

    for i in range(size):
        annoy_index.add_item(i, np.random.rand(5))

    annoy_index.build(10)

    start_time = time.time()
    annoy_index.get_nns_by_vector(np.random.rand(5), 5)
    query_times.append(time.time() - start_time)

# Plot Results
plt.plot(dataset_sizes, query_times, marker='o')
plt.xlabel("Dataset Size")
plt.ylabel("Query Time (seconds)")
plt.title("Scalability: Query Time vs. Dataset Size")
plt.show()
