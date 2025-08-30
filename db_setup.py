# fx_db_setup.py

import sqlite3

# Connect or create new DB
conn = sqlite3.connect('fx_trades.db')
cursor = conn.cursor()

# Drop existing tables (if rerunning script)
cursor.execute('DROP TABLE IF EXISTS trades')
cursor.execute('DROP TABLE IF EXISTS counterparties')

# Create counterparties table
cursor.execute('''
CREATE TABLE counterparties (
    cp_id INTEGER PRIMARY KEY,
    cp_name TEXT,
    region TEXT
)
''')

# Create trades table
cursor.execute('''
CREATE TABLE trades (
    trade_id INTEGER PRIMARY KEY,
    cp_id INTEGER,
    px_type TEXT,      -- Product type: spot, fwd, swap, ndf
    notl REAL,         -- Notional amount
    ccy_pair TEXT,     -- Currency pair
    near_dt TEXT,      -- Near date
    far_dt TEXT,       -- Far date (used only for swaps)
    rate REAL,         -- Executed FX rate
    FOREIGN KEY (cp_id) REFERENCES counterparties(cp_id)
)
''')

# Insert sample counterparties
counterparties = [
    (1, 'Goldman Sachs', 'AMER'),
    (2, 'HSBC', 'EMEA'),
    (3, 'Nomura', 'APAC'),
    (4, 'Deutsche Bank', 'EMEA'),
    (5, 'JP Morgan', 'AMER'),
    (6, 'Barclays', 'EMEA'),
    (7, 'Standard Chartered', 'APAC')
]
cursor.executemany('INSERT INTO counterparties VALUES (?, ?, ?)', counterparties)

# Insert sample FX trades
trades = [
    (101, 1, 'spot', 5_000_000, 'EUR/USD', '2025-08-28', None, 1.1012),
    (102, 2, 'fwd', 12_000_000, 'USD/JPY', '2025-09-10', None, 149.34),
    (103, 3, 'swap', 15_000_000, 'GBP/USD', '2025-09-01', '2026-03-01', 1.2801),
    (104, 4, 'ndf', 7_500_000, 'USD/INR', '2025-08-30', None, 83.45),
    (105, 5, 'swap', 20_000_000, 'USD/CHF', '2025-09-15', '2026-06-15', 0.8922),
    (106, 6, 'spot', 6_000_000, 'USD/CAD', '2025-08-29', None, 1.3256),
    (107, 2, 'fwd', 4_000_000, 'EUR/GBP', '2025-10-01', None, 0.8572),
    (108, 3, 'swap', 8_000_000, 'AUD/USD', '2025-08-20', '2026-02-20', 0.6584),
    (109, 7, 'spot', 9_500_000, 'USD/SGD', '2025-08-27', None, 1.3510),
    (110, 1, 'ndf', 3_000_000, 'USD/KRW', '2025-08-26', None, 1342.2),
    (111, 6, 'swap', 11_000_000, 'USD/MXN', '2025-09-05', '2026-05-05', 16.82),
    (112, 5, 'fwd', 7_000_000, 'USD/BRL', '2025-09-12', None, 5.24)
]
cursor.executemany('INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?)', trades)

conn.commit()
conn.close()
print("✅ FX database created and filled with dummy data (fx_trades.db)")
