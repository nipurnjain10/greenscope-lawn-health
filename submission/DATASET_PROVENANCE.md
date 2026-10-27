# Dataset provenance

The primary file `data/synthetic_lawn_inspections.csv` and the robustness file `data/robustness_scenarios.csv` are synthetic. Every row is a simulated lawn inspection created by `data/generate_synthetic_data.py`.

The files do not contain:

- customer, household, or employee records
- names, telephone numbers, email addresses, or street addresses
- branch or company identifiers
- rows copied from a public dataset
- the latent variables or the continuous score used to define the label

`overall_lawn_health_risk` is a three-way bin of a noisy score. On the primary sample the cuts are the 33rd and 67th percentiles of that score. The robustness sample uses those same cuts. The classes mean Low, Moderate, and High simulated risk. They are not a verified biological diagnosis.

Domain pages listed in `data/source_register.md` were read for context: sunlight, drainage, compaction, soil testing, pH, the limits of a nitrogen soil test, mowing, watering, thatch, and weed identification. They were not the source of the 10,000 rows. FAO EcoCrop's home page was opened. No EcoCrop species table was downloaded.

Nutrient columns are milligrams per kilogram on a scale defined for this project. They are not Rutgers sufficiency classes and they are not fertiliser grades.

Generator version, seeds, score cutoffs, class counts, and SHA-256 fingerprints are in `data/metadata.json`. A later edit to the generator must change `GENERATOR_VERSION` and regenerate the CSV files, the models, and the report.
