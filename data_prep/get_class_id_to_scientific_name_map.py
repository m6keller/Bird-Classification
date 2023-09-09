import json 
import pandas as pd

OUTPUT_PATH = "./metadata/class_id_to_scientific_name_map.json"
METADATA_CSV_PATH = "./metadata/birds.csv"

def main():
    df = pd.read_csv(METADATA_CSV_PATH)
    class_id_to_sci_name = {}
    for i, row in df.iterrows():
        class_id_to_sci_name[row["class id"]] = row["scientific name"]
        
    with open(OUTPUT_PATH, "w") as f:
        json.dump(class_id_to_sci_name, f, indent=2)
        
if __name__ == "__main__":
    main()