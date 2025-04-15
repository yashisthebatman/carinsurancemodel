import sqlite3
import os

# Ensure the database directory exists
os.makedirs('database', exist_ok=True)

# Connect to the database (creates it if it doesn't exist)
conn = sqlite3.connect('database/dl_data.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
CREATE TABLE IF NOT EXISTS driver_licenses (
    license_id TEXT PRIMARY KEY,
    age TEXT,
    gender TEXT,
    driving_experience TEXT,
    education TEXT,
    income TEXT,
    vehicle_ownership INTEGER,
    vehicle_year TEXT,
    married INTEGER,
    children INTEGER,
    speeding_violations INTEGER,
    past_accidents INTEGER
)
''')

# Sample data - 10 fictional driver records
sample_data = [
    ('DL123456789', '26-39', 'male', '10-19y', 'university', 'upper class', 1, 'after 2015', 1, 0, 0, 0),
    ('DL234567890', '40-64', 'female', '20-29y', 'university', 'middle class', 1, 'after 2015', 1, 1, 1, 0),
    ('DL345678901', '16-25', 'male', '0-9y', 'high school', 'working class', 0, 'before 2015', 0, 0, 2, 1),
    ('DL456789012', '65+', 'female', '30+ y', 'university', 'middle class', 1, 'before 2015', 1, 0, 0, 1),
    ('DL567890123', '26-39', 'male', '10-19y', 'high school', 'working class', 1, 'after 2015', 0, 1, 3, 0),
    ('DL678901234', '40-64', 'female', '20-29y', 'university', 'upper class', 1, 'after 2015', 1, 1, 0, 0),
    ('DL789012345', '16-25', 'female', '0-9y', 'high school', 'poverty', 0, 'before 2015', 0, 0, 6, 3),
    ('DL890123456', '26-39', 'male', '10-19y', 'none', 'working class', 1, 'before 2015', 1, 1, 2, 2),
    ('DL901234567', '40-64', 'male', '20-29y', 'university', 'middle class', 1, 'after 2015', 1, 1, 1, 0),
    ('DL012345678', '65+', 'female', '30+ y', 'high school', 'poverty', 0, 'before 2015', 0, 0, 4, 3)
]

# Insert sample data
cursor.executemany('''
INSERT OR REPLACE INTO driver_licenses (
    license_id, age, gender, driving_experience, education, income, 
    vehicle_ownership, vehicle_year, married, children, speeding_violations, past_accidents
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', sample_data)

# Commit changes and close connection
conn.commit()
conn.close()

print("Database created successfully with 10 sample driver records!")