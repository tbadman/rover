
"""
Relative path to sqlite database
"""
DATABASE = './data/db1.sqlite3'

"""
List of assertions for pytest (source: instructions.pdf - excersize I)
Values are expected to be true upon connection to config.DATABASE.
Run assertions using pytest, any failures may indicate connection failure or data corruption
"""
TOTAL_USERS = 64393
TOTAL_USERS_PRIOR = 35826
PERCENT_W_PETS = 80.43
AVG_PETS_USER = 1.501
PERCENT_WELL_W_CATS = 24.85