# Source register

Access date for every page below: 22 October 2026.

These pages are domain guidance. None of them is the 10,000-row table. No row in `data/synthetic_lawn_inspections.csv` or `data/robustness_scenarios.csv` was downloaded from them.

## A. Starting a New Lawn

- Organization: University of Maryland Extension
- URL: https://extension.umd.edu/resource/starting-new-lawn
- Page status: retrieved. The page title and the update line "Updated: March 16, 2026" were visible.
- What was used: site evaluation before planting; full sun described as about 6 or more hours of direct light, and the statement that lawn grasses do not thrive in little direct sun; avoidance of sites that hold water and drain slowly; compaction from foot traffic slowing drainage and resisting rewetting; laboratory soil testing for pH and nutrients rather than a home kit; mildly acidic soil around 6.0–6.8 as favourable, lime considered below 5.5, and sulfur discussed above 7.0; most Maryland lawns described as dominated by tall fescue; cool-season and warm-season grasses, including zoysia and bermudagrass, as establishment options.
- What was not used: seeding rates, sod installation steps, and photographs. No records were downloaded.
- Limitation: the page is Maryland guidance for establishing a lawn, not a set of inspection outcomes and not a risk-labelled dataset.

## B. Maintaining an Established Lawn

- Organization: University of Maryland Extension
- URL: https://extension.umd.edu/resource/maintaining-established-lawn
- Page status: retrieved. Update line visible: "Updated: September 14, 2026".
- What was used: about 1 inch (25 mm) of water per week for turf in active growth, as background rather than a label threshold; drought signs including dull colour and footprinting; the warning against light, frequent irrigation of established lawns; compaction and core aeration, including the point that aeration does not fix a chronically saturated site; the same pH discussion as the establishment page, including a higher upper pH tolerance noted for zoysia and bermudagrass; the instruction not to guess nutrient need and to use a laboratory soil test; thatch thicker than about one-half inch (12.7 mm) as a dethatching screen; mowing that removes no more than about one-third of the leaf.
- What was not used: Maryland fertilizer-law application rates. The prototype does not calculate a fertiliser quantity.
- Limitation: management advice for Maryland lawns, not paired with the features in this project's CSV.

## C. Soil Testing for Home Lawns and Gardens (FS797)

- Organization: Rutgers Cooperative Extension
- Authors named on the page: Joseph Heckman
- URL: https://njaes.rutgers.edu/fs797/
- Page status: retrieved. The page is dated August 2003.
- What was used: a soil test as the starting point for lime and fertiliser decisions; the report contents named on the page (phosphorus, potassium, calcium, magnesium, copper, manganese, zinc, and pH); separate samples for areas that differ; the warning that a test is only as good as the sample.
- What was not used: no Rutgers laboratory file was downloaded. Nitrogen is not in the result list quoted on that page.
- Limitation: a sampling fact sheet, not a table of lawn-health classes.

## D. Fertilizing the Home Lawn (FS633)

- Organization: Rutgers Cooperative Extension
- Authors named on the page: Joseph Heckman and James A. Murphy
- URL: https://njaes.rutgers.edu/FS633/
- Page status: retrieved. The page is dated January 2003.
- What was used: the statement that an attractive lawn needs nutrients and a slightly acid pH; the statement that a soil test has not been developed to measure turfgrass nitrogen need; the statement that phosphorus, potassium, and lime for an established lawn should follow a soil test; the caution that some lawns receive more phosphorus and lime than is useful; the statement that a pH near 6.5 is favourable and that raising pH above 6.5 has no general benefit and can reduce availability of iron, manganese, and zinc.
- What was not used: the page's pounds-per-1,000-square-feet fertiliser tables. Those rates are not implemented, and the project's mg/kg columns are not claimed to be the categories in those tables.
- Limitation: New Jersey home-lawn guidance. Nutrient units in the CSV are a project scale.

## E. Your Lawn and Its Care (FS102)

- Organization: Rutgers Cooperative Extension
- Author named on the page: James A. Murphy
- URL: https://njaes.rutgers.edu/fs102/
- Page status: retrieved. The page is dated October 2023.
- What was used: grass selection for the site; mowing height and the one-third rule; returning clippings; watering symptoms and deep, infrequent irrigation; thatch over one-half inch; weed prevention by a dense lawn and the requirement to identify a weed before any herbicide decision; the point that a healthy lawn tolerates some insects and disease and that diagnosis needs expertise.
- What was not used: no disease or insect is predicted by the model. No herbicide or pesticide rate is recommended.
- Limitation: a care fact sheet, not an observational dataset.

## F. FAO EcoCrop

- Organization: Food and Agriculture Organization of the United Nations
- URL: https://ecocrop.apps.fao.org/ecocrop/srv/en/home
- Page status: the home page was retrieved. It describes a tool that can match crops to temperature, rainfall, light, soil texture, depth, pH, salinity, and fertility, and that can look up a named crop.
- What was used: only as a reminder that environmental requirements are crop-specific. No EcoCrop species record, climate table, or survival observation was downloaded.
- What was not used: salinity, soil depth, and any species-specific EcoCrop number. Grass-specific shade tolerances are not taken from EcoCrop.
- Limitation: EcoCrop is not a turfgrass survival dataset, and this project did not query it for individual grasses.

## Candidate datasets that were not used

A web search on 22 October 2026 found public or semi-public turf and lawn material. None of it was downloaded, and none of it is a dependency of the generator.

- Micah Woods, 2016 MLSN paper data folder on GitHub (`https://github.com/micahwoods/2016_mlsn_paper/tree/master/data`). The folder listing describes soil-nutrient analyses of professionally managed turf, largely golf turf. A licence file was not retrieved, so the rows were not reused. The task is also different: nutrient analyses, not a three-class lawn-health label.
- Stefanowicz and colleagues, urban roadside lawn soil and vegetation, RepOD, DOI `10.18150/W3CKSR`. The repository text returned by the search stated an embargo until 2 June 2027. The files were not downloaded.
- US Department of Agriculture, Agricultural Research Service, turfgrass soil-carbon meta-analysis (`http://catalog.data.gov/dataset/turfgrass-soil-carbon-change-through-time-raw-data-and-code`). The catalogue record describes soil-carbon studies, not inspection-level health-risk labels. The files were not downloaded.

## Project files that are synthetic

| File | What it is |
| --- | --- |
| `data/synthetic_lawn_inspections.csv` | 10,000 simulated inspections produced by `data/generate_synthetic_data.py` |
| `data/robustness_scenarios.csv` | 2,000 simulated inspections from a second seed and a shifted scenario |
| `data/metadata.json` | Generator version, seeds, score cutoffs, and file fingerprints |

Regenerate both CSV files with `python data/generate_synthetic_data.py`. The command does not call an external data service.
