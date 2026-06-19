import os
import pandas as pd

def main():
    xlsx_path = "/var/www/analiza-dicom/Library/Mobile Documents/com~apple~CloudDocs/AngioPy/AngioPy.xlsx"
        
    print(f"Checking Excel file at: {xlsx_path}")
    if os.path.exists(xlsx_path):
        try:
            df = pd.read_excel(xlsx_path)
            print(f"Total rows in Excel: {len(df)}")
            df_1301 = df[df["Patient ID"].astype(str).str.startswith("1301")]
            print(f"Total rows for 1301 in Excel: {len(df_1301)}")
            if len(df_1301) > 0:
                print(df_1301[["Patient ID", "DICOM Name", "Phase", "Vessel", "AHA Segment"]].head(20))
        except Exception as e:
            print(f"Error reading Excel: {e}")
    else:
        print("Excel file not found!")

if __name__ == "__main__":
    main()
