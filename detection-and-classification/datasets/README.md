# Datasets Description

## Objective
The purpose of this document is to describe the datasets used in this project. Multiple datasets were selected in order to cover different types of road-related elements such as obstacles, road surface degradations, debris, and traffic signs. Each dataset focuses on a specific category of objects or events, allowing a clear separation between different road scenarios and reducing ambiguity during data analysis.

---

## Datasets Used

### Road Damage Dataset
**Objective:** Detect road surface degradations.
**Content:**
- potholes
- longitudinal cracks
- lateral cracks
- surface damages
**Description:** This dataset contains images of damaged road surfaces with annotations identifying different types of degradations. It is used to represent the condition of the road itself.
**Source:** Kaggle – Road Damage Dataset.

---

### Lost and Found Dataset
**Objective:** Detect rare and unexpected obstacles on the road.
**Content:**
- abandoned objects
- uncommon obstacles
- unexpected hazardous elements
**Description:** This dataset focuses on rare situations that are usually underrepresented in standard road datasets. It is useful for studying robustness in unusual scenarios.
**Source:** Lost and Found Dataset (autonomous driving research).

---

### Road Debris Dataset
**Objective:** Detect debris and physical objects on the road.
**Content:**
- debris
- blocks
- miscellaneous road objects
**Description:** This dataset represents objects that may obstruct the road and affect driving safety.
**Source:** Kaggle – Road Debris Dataset.

---

### Road Work Sign Dataset
**Objective:** Detect road work and construction zone elements.
**Content:**
- road work signs
- cones
- warning indicators
**Description:** This dataset is used to identify construction areas and temporary road configurations.
**Source:** Public road work sign datasets.

---

### Stop Sign Dataset
**Objective:** Detect stop signs.
**Content:**
- stop sign
**Description:** This dataset focuses on a single critical traffic sign to ensure reliable detection.
**Source:** Public traffic sign datasets.

---

### Speed Bumps Dataset
**Objective:** Detect speed bumps on the road.
**Content:**
- speed bumps
- raised road sections
**Description:** This dataset represents road elements that require speed adaptation by the vehicle.
**Source:** Public speed bump datasets.

---

## Data Format
All datasets are organized using the YOLO format:
- images and labels are separated into training, validation, and test sets
- annotations are stored in text files
- each dataset includes a `data.yaml` configuration file

This unified format ensures consistency across all datasets.
