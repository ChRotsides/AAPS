from generatePatientData import GenPatientData

# Path: TrainingModels\trainModels.py
options=[#"adolescent#001",
        # "adolescent#002",
        # "adolescent#007",
        # "adult#001",
        # "adult#002",
        # "adult#009",
        "child#002",
        "child#006",
        "child#008",]
for option in options:
    print("Generating data for:",option)
    env=GenPatientData(patient_name=option)
    env.run(env.get_loops(days=14,hours=0))
    env.save_data()
