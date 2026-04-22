import sqlite3, random, os

random.seed(42)

DB_PATH = '../data/ecommerce_test.db'
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

c.execute("""CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT NOT NULL,
    parent_category TEXT
)""")
cats = [
    (1, 'Electronics', None), (2, 'Clothing', None), (3, 'Home & Garden', None),
    (4, 'Books', None), (5, 'Sports', None), (6, 'Laptops', 'Electronics'),
    (7, 'Phones', 'Electronics'), (8, 'Men', 'Clothing'), (9, 'Women', 'Clothing'),
    (10, 'Fiction', 'Books'),
]
c.executemany("INSERT INTO categories VALUES (?,?,?)", cats)

c.execute("""CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category_id INTEGER REFERENCES categories(category_id),
    unit_price REAL NOT NULL,
    stock_quantity INTEGER NOT NULL,
    brand TEXT,
    weight_kg REAL,
    created_date TEXT
)""")
brands = ['Samsung', 'Apple', 'Nike', 'Adidas', 'Sony', 'LG', 'Dell', 'HP', 'Penguin', 'Levi']
products = []
for i in range(1, 201):
    cat = random.choice(cats)
    price = round(random.uniform(5, 999), 2)
    products.append((i, f"Product_{i}", cat[0], price, random.randint(0, 500),
                      random.choice(brands), round(random.uniform(0.1, 15), 1),
                      f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}"))
c.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?)", products)

c.execute("""CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    first_name TEXT, last_name TEXT,
    email TEXT, city TEXT, state TEXT, country TEXT,
    registration_date TEXT, date_of_birth TEXT,
    loyalty_tier TEXT
)""")
cities = [('New York','NY','US'),('Los Angeles','CA','US'),('Chicago','IL','US'),
          ('Houston','TX','US'),('London','','UK'),('Toronto','ON','CA'),
          ('Sydney','NSW','AU'),('Berlin','','DE'),('Tokyo','','JP'),('Paris','','FR')]
tiers = ['Bronze','Silver','Gold','Platinum']
customers = []
for i in range(1, 1001):
    city = random.choice(cities)
    yob = random.randint(1955, 2005)
    customers.append((i, f"First{i}", f"Last{i}", f"user{i}@email.com",
                       city[0], city[1], city[2],
                       f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                       f"{yob}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                       random.choice(tiers)))
c.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?,?,?,?)", customers)

c.execute("""CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    order_date TEXT NOT NULL,
    ship_date TEXT,
    status TEXT NOT NULL,
    shipping_method TEXT,
    total_amount REAL
)""")
statuses = ['completed','completed','completed','completed','shipped','processing','cancelled','returned']
ship_methods = ['Standard','Express','Overnight','Free']
orders = []
for i in range(1, 5001):
    cust = random.randint(1, 1000)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    st = random.choice(statuses)
    orders.append((i, cust, f"2025-{month:02d}-{day:02d}",
                    f"2025-{month:02d}-{min(day+random.randint(1,5),28):02d}" if st in ('completed','shipped') else None,
                    st, random.choice(ship_methods), round(random.uniform(10, 2000), 2)))
c.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?)", orders)

c.execute("""CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    discount_pct REAL DEFAULT 0
)""")
items = []
item_id = 1
for oid in range(1, 5001):
    for _ in range(random.randint(1, 4)):
        pid = random.randint(1, 200)
        qty = random.randint(1, 5)
        items.append((item_id, oid, pid, qty,
                       products[pid-1][3], round(random.uniform(0, 0.3), 2)))
        item_id += 1
c.executemany("INSERT INTO order_items VALUES (?,?,?,?,?,?)", items)

c.execute("""CREATE TABLE reviews (
    review_id INTEGER PRIMARY KEY,
    product_id INTEGER REFERENCES products(product_id),
    customer_id INTEGER REFERENCES customers(customer_id),
    rating INTEGER NOT NULL,
    review_text TEXT,
    review_date TEXT
)""")
reviews = []
for i in range(1, 3001):
    reviews.append((i, random.randint(1,200), random.randint(1,1000),
                     random.randint(1,5), f"Review text {i}",
                     f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}"))
c.executemany("INSERT INTO reviews VALUES (?,?,?,?,?,?)", reviews)

c.execute("""CREATE TABLE suppliers (
    supplier_id INTEGER PRIMARY KEY,
    supplier_name TEXT, contact_email TEXT, country TEXT, rating REAL
)""")
suppliers = []
for i in range(1, 51):
    suppliers.append((i, f"Supplier_{i}", f"supplier{i}@biz.com",
                       random.choice(['US','CN','DE','JP','KR','TW','IN']),
                       round(random.uniform(3, 5), 1)))
c.executemany("INSERT INTO suppliers VALUES (?,?,?,?,?)", suppliers)

c.execute("""CREATE TABLE product_suppliers (
    product_id INTEGER REFERENCES products(product_id),
    supplier_id INTEGER REFERENCES suppliers(supplier_id),
    supply_price REAL, lead_time_days INTEGER,
    PRIMARY KEY (product_id, supplier_id)
)""")
ps = []
for pid in range(1, 201):
    for sid in random.sample(range(1, 51), random.randint(1, 3)):
        ps.append((pid, sid, round(products[pid-1][3]*random.uniform(0.4, 0.7), 2),
                    random.randint(3, 30)))
c.executemany("INSERT INTO product_suppliers VALUES (?,?,?,?)", ps)

conn.commit()
conn.close()

conn = sqlite3.connect(DB_PATH)
for tbl in ['categories','products','customers','orders','order_items','reviews','suppliers','product_suppliers']:
    cnt = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    print(f"  {tbl:20s}: {cnt:>6} rows")
conn.close()
print(f"\nCreated: {DB_PATH}")
