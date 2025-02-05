# digital_twin/data_collector.py

import json
from sqlalchemy.orm import sessionmaker
from digital_twin.data_storage import engine, SimulationResult, init_db
from data_generation.simulator import run_simulation

# Initialize the database (create tables if they don't exist)
init_db()

SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

def collect_data(num_simulations=20000):
    """
    Runs the simulation multiple times and saves results to the database.
    Reward fields have been removed.
    """
    for i in range(num_simulations):
        results = run_simulation()

        # Extract fields from each stage; rewards removed.
        sim_entry = SimulationResult(
            # Mixing stage
            mixing_status=results.get('mixing', {}).get('status'),
            mixing_operator=results.get('mixing', {}).get('operator'),
            mixing_time=results.get('mixing', {}).get('mixing_time'),
            mixing_speed=results.get('mixing', {}).get('mixing_speed'),
            uniformity_index=results.get('mixing', {}).get('uniformity_index'),
            mixing_log=json.dumps(results.get('mixing', {}).get('log', {})),
            
            # Granulation stage
            granulation_status=results.get('granulation', {}).get('status'),
            granulation_operator=results.get('granulation', {}).get('operator'),
            granulation_time=results.get('granulation', {}).get('granulation_time'),
            binder_rate=results.get('granulation', {}).get('binder_rate'),
            granule_density=results.get('granulation', {}).get('granule_density'),
            granulation_log=json.dumps(results.get('granulation', {}).get('log', {})),
            
            # Drying stage
            drying_status=results.get('drying', {}).get('status'),
            drying_operator=results.get('drying', {}).get('operator'),
            drying_temp=results.get('drying', {}).get('drying_temp'),
            moisture_content=results.get('drying', {}).get('moisture_content'),
            drying_log=json.dumps(results.get('drying', {}).get('log', {})),
            
            # Compression stage
            compression_status=results.get('compression', {}).get('status'),
            compression_operator=results.get('compression', {}).get('operator'),
            comp_pressure=results.get('compression', {}).get('comp_pressure'),
            tablet_hardness=results.get('compression', {}).get('tablet_hardness'),
            weight_variation=results.get('compression', {}).get('weight_variation'),
            dissolution=results.get('compression', {}).get('dissolution'),
            yield_percent=results.get('compression', {}).get('yield_percent'),
            compression_log=json.dumps(results.get('compression', {}).get('log', {})),
            
            # Overall simulation result
            final_status=results.get('final_status'),
            failure_reason=(results.get('mixing', {}).get('reason') or
                            results.get('granulation', {}).get('reason') or
                            results.get('drying', {}).get('reason') or
                            results.get('compression', {}).get('reason') or '')
        )

        session.add(sim_entry)
        session.commit()

        print(f"Simulation {i+1}/{num_simulations} stored in DB.")

    session.close()

if __name__ == "__main__":
    collect_data()
