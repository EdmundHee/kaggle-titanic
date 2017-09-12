import psycopg2

class ImportData:

    def __init__(self, options):
        self.options = options
        self.data_dir = "../data/"

    def connect_db(self):
        try:
            conn = psycopg2.connect("dbname='kaggle' user='postgres' host='localhost' password='postgres'")
            
        except:
            print("Not able to connect to db")

options = {}
import_data = ImportData(options)
import_data.connect_db()
