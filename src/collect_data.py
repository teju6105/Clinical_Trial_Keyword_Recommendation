import requests
import pandas as pd
import time
from pathlib import Path

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DATA_DIR / "clinical_trials.csv"

MAX_STUDIES = 1000
PAGE_SIZE = 100


def collect_trials():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    next_page_token = None

    print("Connecting to ClinicalTrials.gov...")
    print(f"Target: Up to {MAX_STUDIES} clinical trial records")

    while len(records) < MAX_STUDIES:
        remaining = MAX_STUDIES - len(records)
        params = {
            "format": "json",
            "pageSize": min(PAGE_SIZE, remaining),
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        try:
            response = requests.get(
                BASE_URL,
                params=params,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

        except requests.RequestException as error:
            print(f"API request failed: {error}")
            break

        studies = data.get("studies", [])

        if not studies:
            print("No additional records returned.")
            break

        for study in studies:
            protocol = study.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            status = protocol.get("statusModule", {})
            design = protocol.get("designModule", {})
            description = protocol.get("descriptionModule", {})
            conditions_module = protocol.get("conditionsModule", {})
            arms_module = protocol.get("armsInterventionsModule", {})

            interventions = arms_module.get("interventions", []) or []
            intervention_names = [
                item.get("name", "")
                for item in interventions
                if item.get("name")
            ]

            records.append({
                "nct_id": identification.get("nctId", ""),
                "title": identification.get("briefTitle", ""),
                "official_title": identification.get("officialTitle", ""),
                "status": status.get("overallStatus", ""),
                "study_type": design.get("studyType", ""),
                "phase": ", ".join(design.get("phases", []) or []),
                "conditions": "; ".join(
                    conditions_module.get("conditions", []) or []
                ),
                "keywords": "; ".join(
                    conditions_module.get("keywords", []) or []
                ),
                "brief_summary": description.get("briefSummary", "") or "",
                "interventions": "; ".join(intervention_names),
            })

        print(f"Collected {len(records)} / {MAX_STUDIES} records")

        next_page_token = data.get("nextPageToken")

        if not next_page_token:
            print("Reached the end of available results.")
            break

        time.sleep(0.2)

    if not records:
        print("No records collected. Check your internet connection.")
        return

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["nct_id"])
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 50)
    print("DATA COLLECTION COMPLETED")
    print("=" * 50)
    print(f"Records saved: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nFirst five records:")
    print(
        df[["nct_id", "title", "conditions", "keywords"]]
        .head()
        .to_string(index=False)
    )

    print("\nMissing values per column:")
    print(df.isnull().sum())


if __name__ == "__main__":
    collect_trials()
