# digital_twin/query_data.py

from sqlalchemy.orm import sessionmaker
from digital_twin.data_storage import engine, SimulationResult, init_db
import json

# Initialize the database (create tables if they don't exist)
init_db()

SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

def query_simulations(limit=10):
    """
    Query the first `limit` simulation records and print key details.
    """
    results = session.query(SimulationResult).limit(limit).all()
    for result in results:
        print(f"Run ID: {result.id}")
        print(f"  Final Status: {result.final_status}")
        print(f"  Mixing: status={result.mixing_status}")
        print(f"  Granulation: status={result.granulation_status}")
        print(f"  Drying: status={result.drying_status}")
        print(f"  Compression: status={result.compression_status}")
        print(f"  Failure Reason: {result.failure_reason}")
        print("  ---")
        # Optionally, deserialize logs:
        mixing_log = json.loads(result.mixing_log) if result.mixing_log else {}
        print(f"  Mixing Log: {mixing_log}")

        # Print all:
        print(result.__dict__)

    session.close()

if __name__ == "__main__":
    query_simulations()
