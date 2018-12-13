"""
Test module for database connection.
Assertions are located in config.py (provided by instructions.pdf)

Since query results were given in Excersise I, I decided to use
the opportunity to build 'Exploring the Database' into a unit-test
module for asserting an accurate connection and complete dataset.

Usage: run 'pytest' from CLI (requires pytest)

December 2018
"""
import pytest
from db_interact import DBInteract
import config

@pytest.fixture(scope="module")
def db_connection():
    """pytest fixture for db connection"""
    return DBInteract(config.DATABASE)

def test_total_users(db_connection):
    """Test user count in people_person is equal to config.TOTAL_USERS"""
    query = """
            SELECT count(*) as result
            FROM people_person
            """
    assert db_connection.simple_query(query) == config.TOTAL_USERS

def test_total_users_prior(db_connection):
    """Test users who joined prior to 2018-02-03 is equal to config.TOTAL_USERS_PRIOR"""
    query = """
            SELECT count(*) as result
            FROM people_person
            WHERE date_joined < '2018-02-03'
            """
    assert db_connection.simple_query(query) == config.TOTAL_USERS_PRIOR

def test_percent_users_pets(db_connection):
    """Test percent of users with pets is equal to config.PERCENT_W_PETS"""
    query = """
            SELECT ROUND(CAST(count(*) as float)/(SELECT count(*) FROM people_person)*100.,2) as result
            FROM people_person per
            WHERE (SELECT exists (SELECT 1 FROM pets_pet pet WHERE pet.owner_id = per.id))
            """
    assert db_connection.simple_query(query) == config.PERCENT_W_PETS

def test_avg_pets(db_connection):
    """Test average number of pets per user is equal to config.AVG_PETS_USER"""
    query = """
            SELECT ROUND(CAST(sum(t.n_pets) as float)/count(*),3) as result
            FROM (
                    SELECT per.id, count(*) as n_pets
                    FROM people_person per
                    JOIN pets_pet pet on (per.id = pet.owner_id)
                    GROUP BY per.id
            ) t
            """
    assert db_connection.simple_query(query) == config.AVG_PETS_USER

def test_percent_well_cats(db_connection):
    """Test percent of pets that do well with cats is equal to config.PERCENT_WELL_W_CATS"""
    query = """
            SELECT ROUND(CAST(count(*) as float)/(SELECT count(*) FROM pets_pet)*100.,2) as result
            FROM pets_pet
            WHERE plays_cats
            """
    assert db_connection.simple_query(query) == config.PERCENT_WELL_W_CATS
