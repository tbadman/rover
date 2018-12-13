"""
SQLite3 database interaction class for petcare_analysis.
Provides a connection object and a simple query method.

December 2018
"""

import sqlite3
import pandas as pd

class DBInteract(): # pylint: disable=too-few-public-methods
    """SQLite DB connection object"""

    def __init__(self, database):
        """Build database connection. Requires path to DB"""

        self.database = database
        self.conn = sqlite3.connect(self.database)

    def simple_query(self, query):
        """Simple framework for 'single result' queries"""

        query_results = pd.read_sql_query(query, self.conn)

        if 'result' in query_results.columns and len(query_results.index) == 1:
            return query_results.iloc[0]['result'] # pylint: disable=no-member

        print('Invalid simple_query transaction encountered...')
        return None
