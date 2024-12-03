import simpy
import random
import numpy as np
import pandas as pd

# Constants
NUM_LAB_RESOURCES = 12  # Maximum simultaneous lab procedures that can be conducted.
HOLDING_BAY_OPTIONS = range(11, 23)  # Possible numbers of holding bays (decision variable: 11 to 22).
CLOSING_TIME_OPTIONS = [20, 22, 24]  # Possible holding bay closing times: 8pm (20:00), 10pm (22:00), midnight (24:00).

# Patient arrival rates (patients per hour) for different time intervals and procedures.
ARRIVAL_RATES = {
    "CATH": {  # Arrival rates for CATH patients.
        "6am_to_10am": 10,  
        "10am_to_2pm": 8,   
        "2pm_to_6pm": 5,    
        "6pm_to_6am": 0  # No arrivals outside of 6am-6pm.
    },
    "EP": {  # Arrival rates for EP patients.
        "6am_to_10am": 15,  
        "10am_to_2pm": 10,  
        "2pm_to_6pm": 8,    
        "6pm_to_6am": 0  # No arrivals outside of 6am-6pm.
    }
}

# Time distributions for different stages of the patient journey.
PREP_TIME_MEAN = 20  # Mean pre-procedure preparation time in minutes (exponential distribution).
CATH_PROCEDURE_TIME = lambda: random.expovariate(1 / 18.8)  # CATH procedure time in minutes (exponential).
EP_PROCEDURE_TIME = lambda: random.expovariate(1 / 14.1)    # EP procedure time in minutes (exponential).
RECOVERY_TIME = lambda: max(0, random.gauss(49.4, 9.97))    # Recovery time in minutes (Gaussian distribution).

# Cost weights for various events in the simulation.
RENEGE_COST = 100  # Cost incurred when a patient leaves due to long wait times.
CANCELLATION_COST = 200  # Cost for canceling a procedure due to operational constraints.
IDLE_BAY_COST = 0.167  # Cost per minute for idle holding bays.
TRANSFERRED_COST = 20  # Cost for transferring a patient to another facility.

# Function to reset simulation metrics.
def reset_metrics():
    """Initialize/reset metrics for tracking simulation results."""
    return {
        "total_patients": 0,  # Total number of patients processed.
        "cancellations": 0,  # Number of procedure cancellations.
        "reneged": 0,  # Number of patients who left due to long wait times.
        "transferred": 0,  # Number of patients transferred due to holding bay closure.
        "idle_bay_time": 0,  # Total idle time of holding bays.
        "used_lab_time": 0,  # Total time lab resources were in use.
        "patient_wait_time": [],  # Total wait times of all patients (from arrival to completion).
        "pre_procedure_wait_times": [],  # Wait times before pre-procedure preparation.
        "post_procedure_wait_times": []  # Wait times before recovery after the procedure.
    }

# Generator function for patient arrivals.
def patient_generator(env, lab, holding_bays, arrival_rate, patient_type, closing_time, metrics):
    """Simulates the arrival of patients based on a Poisson process."""
    while True:
        current_hour = int(env.now // 60) % 24  # Determine the current hour of the day.
        if current_hour < 6 or current_hour >= 18:  # No arrivals outside 6am-6pm.
            yield env.timeout(60)  # Skip an hour if outside arrival times.
            continue

        # Determine the appropriate arrival rate based on the time of day.
        if current_hour < 10:  
            rate_key = "6am_to_10am"
        elif current_hour < 14:  
            rate_key = "10am_to_2pm"
        else:  
            rate_key = "2pm_to_6pm"

        # Generate the number of arrivals using a Poisson arrival rate distribution.
        num_arrivals = np.random.poisson(arrival_rate[rate_key]) #Using the arrival rate that was found using the Excel data and input analyzer
        interarrival_time = 60 / arrival_rate[rate_key]  # Average time between arrivals.

        for _ in range(num_arrivals):  # Process each arriving patient.
            yield env.timeout(interarrival_time)  # Wait for the next patient arrival.
            metrics["total_patients"] += 1
            env.process(patient(env, lab, holding_bays, patient_type, closing_time, metrics))  # Start patient process.

# Process for individual patients.
def patient(env, lab, holding_bays, patient_type, closing_time, metrics):
    """Simulates the journey of a patient through the clinic."""
    arrival_time = env.now  # Record the time of arrival.

    # Step 1: Request a holding bay for pre-procedure preparation.
    request_time = env.now
    with holding_bays.request(priority=1) as prep_request:  # Priority 1 for pre-procedure patients.
        result = yield prep_request | env.timeout(60)  # Wait for up to 60 minutes for a bay.
        if prep_request not in result:  # Patient reneges if no bay is available within 60 minutes.
            metrics["reneged"] += 1
            return

        # Calculate and record the pre-procedure wait time.
        wait_time = env.now - request_time
        metrics["pre_procedure_wait_times"].append(wait_time)

        # Simulate the preparation phase.
        prep_time = random.expovariate(1 / PREP_TIME_MEAN)  # Preparation time from exponential distribution.
        yield env.timeout(prep_time)

        # Step 2: Request a lab resource for the procedure.
        lab_start_time = env.now
        with lab.request() as lab_request:  # Request a lab resource.
            result = yield lab_request
            if lab_start_time / 60 >= closing_time:  # Procedure is canceled if lab is closed.
                metrics["cancellations"] += 1
                return

            # Simulate the procedure.
            procedure_time = CATH_PROCEDURE_TIME() if patient_type == "CATH" else EP_PROCEDURE_TIME()
            yield env.timeout(procedure_time)
            metrics["used_lab_time"] += env.now - lab_start_time  # Track lab usage time.

        # Step 3: Recovery phase in a holding bay.
        recovery_request_time = env.now
        with holding_bays.request(priority=0) as recovery_request:  # Priority 0 for recovery patients.
            result = yield recovery_request | env.timeout(100000)  # Wait indefinitely for a recovery bay.
            if recovery_request not in result:  # Patient is transferred if no bay is available.
                metrics["transferred"] += 1
                return

            # Record post-procedure wait time.
            post_wait_time = env.now - recovery_request_time
            metrics["post_procedure_wait_times"].append(post_wait_time)

            # Simulate recovery.
            recovery_time = RECOVERY_TIME()
            recovery_end_time = recovery_request_time + recovery_time
            while env.now < recovery_end_time:  # Ensure recovery completes before bay closure.
                if env.now >= closing_time * 60:  # Transfer patient if recovery exceeds closing time.
                    metrics["transferred"] += 1
                    return
                yield env.timeout(1)  # Wait incrementally to check closure condition.

            # Record total patient wait time (from arrival to recovery completion).
            metrics["patient_wait_time"].append(env.now - arrival_time)

# Simulation function for the clinic.
def simulate_clinic(holding_bays, closing_time):
    """Runs the simulation for a single day and records performance metrics."""
    env = simpy.Environment()  # Initialize simulation environment.
    lab = simpy.Resource(env, NUM_LAB_RESOURCES)  # Define lab with limited resources.
    holding_bays_resource = simpy.PriorityResource(env, capacity=holding_bays)  # Define holding bay with priority access.
    metrics = reset_metrics()  # Reset metrics for the current simulation.

    # Start patient generators for both CATH and EP patient types.
    for patient_type, rates in ARRIVAL_RATES.items():
        env.process(patient_generator(env, lab, holding_bays_resource, rates, patient_type, closing_time, metrics))

    env.run(until=24 * 60)  # Run the simulation for a full day (1440 minutes).

    # Calculate idle bay time based on unused capacity.
    total_bay_capacity = holding_bays * (closing_time - 6) * 60  # Total possible bay usage time.
    used_bay_time = sum(metrics["patient_wait_time"])  # Actual bay usage time.
    metrics["idle_bay_time"] = max(0, total_bay_capacity - used_bay_time)  # Idle time is the unused capacity.

    # Calculate average wait times for performance analysis.
    metrics["avg_pre_procedure_wait_time"] = (
        np.mean(metrics["pre_procedure_wait_times"]) if metrics["pre_procedure_wait_times"] else 0
    )
    metrics["avg_post_procedure_wait_time"] = (
        np.mean(metrics["post_procedure_wait_times"]) if metrics["post_procedure_wait_times"] else 0
    )

    return metrics

# Simulation runner for different configurations.
def run_simulations():
    """Runs the simulation for all combinations of holding bays and closing times."""
    results = []  # Initialize list to store results.

    # Loop through all combinations of holding bay counts and closing times.
    for holding_bays in HOLDING_BAY_OPTIONS:
        for closing_time in CLOSING_TIME_OPTIONS:
            metrics = simulate_clinic(holding_bays, closing_time)  # Run the simulation for the current configuration.
            total_cost = (
                metrics["reneged"] * RENEGE_COST +
                metrics["cancellations"] * CANCELLATION_COST +
                metrics["idle_bay_time"] * IDLE_BAY_COST +
                metrics["transferred"] * TRANSFERRED_COST
            )
            results.append({  # Record the results.
                "Holding Bays": holding_bays,
                "Closing Time": closing_time,
                "Total Patients": metrics["total_patients"],
                "Cancellations": metrics["cancellations"],
                "Reneged": metrics["reneged"],
                "Transferred": metrics["transferred"],
                "Idle Bay Time": metrics["idle_bay_time"],
                "Used Lab Time": metrics["used_lab_time"],  # Total lab usage time.
                "Average Pre-Procedure Wait Time": metrics["avg_pre_procedure_wait_time"],
                "Average Post-Procedure Wait Time": metrics["avg_post_procedure_wait_time"],
                "Total Cost": total_cost
            })

    return pd.DataFrame(results)  # Return results as a DataFrame.

# Execute and print results of simulations.
df_results = run_simulations()

# Configure pandas to display the entire DataFrame for detailed analysis.
pd.set_option('display.max_rows', None)  # Show all rows.
pd.set_option('display.max_columns', None)  # Show all columns.

print(df_results)  # Display the results DataFrame.

# Reset pandas display options after printing.
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

# Identify and display the optimal configuration based on minimum total cost.
optimal_model = df_results.loc[df_results["Total Cost"].idxmin()]
print("Optimal Model Configuration:")
print(optimal_model)
