# digital_twin/data_storage.py

import os
from sqlalchemy import create_engine, Column, Integer, Float, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Ensure the database file is created in the proper directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_FILE = os.path.join(BASE_DIR, "simulation_data.db")

# Initialize SQLAlchemy
Base = declarative_base()
engine = create_engine(f"sqlite:///{DATABASE_FILE}")
SessionLocal = sessionmaker(bind=engine)

# Define a table to store simulation results (reward fields removed)
class SimulationResult(Base):
    __tablename__ = "simulation_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Mixing stage fields
    mixing_status = Column(String)
    mixing_operator = Column(String)
    mixing_time = Column(Float)
    mixing_speed = Column(Float)
    uniformity_index = Column(Float)
    mixing_log = Column(Text)  # JSON serialized

    # Granulation stage fields
    granulation_status = Column(String)
    granulation_operator = Column(String)
    granulation_time = Column(Float)
    binder_rate = Column(Float)
    granule_density = Column(Float)
    granulation_log = Column(Text)  # JSON serialized

    # Drying stage fields
    drying_status = Column(String)
    drying_operator = Column(String)
    drying_temp = Column(Float)
    moisture_content = Column(Float)
    drying_log = Column(Text)  # JSON serialized

    # Compression stage fields
    compression_status = Column(String)
    compression_operator = Column(String)
    comp_pressure = Column(Float)
    tablet_hardness = Column(Float)
    weight_variation = Column(Float)
    dissolution = Column(Float)
    yield_percent = Column(Float)
    compression_log = Column(Text)  # JSON serialized

    # Overall simulation outcome
    final_status = Column(String)
    failure_reason = Column(String)

# Create tables
def init_db():
    Base.metadata.create_all(engine)

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
