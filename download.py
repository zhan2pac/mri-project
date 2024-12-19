import asyncio
import aiohttp
import os
import pandas as pd
import random

def read_patient_ids_from_excel(file_path):
    df = pd.read_excel(file_path)
    return df[['Anonymized_Patient', 'age_old', 'main_part']].dropna().to_dict(orient='records')

async def fetch_json(session, url, auth):
    async with session.get(url, auth=auth) as response:
        if response.status != 200:
            print(f"Failed to fetch {url}: {response.status} {await response.text()}")
            return None
        return await response.json()

async def download_file(session, url, file_path, auth, retries=3):
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping download.")
        return
    for attempt in range(retries):
        try:
            async with session.get(url, auth=auth) as response:
                if response.status != 200:
                    print(f"Failed to download file from {url}: {response.status} {await response.text()}")
                    return
                with open(file_path, "wb") as f:
                    f.write(await response.read())
                return
        except aiohttp.ClientError as e:
            print(f"Error downloading file from {url}: {e}")
            if attempt < retries - 1:
                print(f"Retrying ({attempt + 1}/{retries})...")
                await asyncio.sleep(1) 
            else:
                print(f"Failed to download file from {url} after {retries} attempts.")
                return

def extract_t1_t2_info(main_part):
    t1_t2_info = []
    if 'Т1' in main_part:
        t1_t2_info.append('T1')
    if 'Т2' in main_part:
        t1_t2_info.append('T2')
    return ', '.join(t1_t2_info)

async def process_patient(session, server_url, patient, output_dir, auth, excel_data):
    patient_data = await fetch_json(session, f"{server_url}/patients/{patient}", auth)
    if not patient_data:
        return None

    patient_id = patient_data['MainDicomTags'].get('PatientID')

    print(f"Processing data for patient: {patient_id}")
    patient_dir = os.path.join(output_dir, patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    studies = await fetch_json(session, f"{server_url}/patients/{patient}/studies", auth)
    if not studies:
        return None

    for study_id in studies:
        study_dir = os.path.join(patient_dir, study_id['ID'])
        os.makedirs(study_dir, exist_ok=True)

        series = await fetch_json(session, f"{server_url}/studies/{study_id['ID']}/series", auth)
        if not series:
            continue

        for series_id in series:
            series_dir = os.path.join(study_dir, series_id['ID'])
            os.makedirs(series_dir, exist_ok=True)

            instances = await fetch_json(session, f"{server_url}/series/{series_id['ID']}/instances", auth)
            if not instances:
                continue

            for instance_id in instances:
                dicom_path = os.path.join(series_dir, f"{instance_id['ID']}.dcm")
                dicom_preview_path = os.path.join(series_dir, f"{instance_id['ID']}.png")

                print(f"Processing instance {instance_id['ID']}")
                await download_file(session, f"{server_url}/instances/{instance_id['ID']}/file", dicom_path, auth)
                await download_file(session, f"{server_url}/instances/{instance_id['ID']}/preview", dicom_preview_path, auth)

    excel_patient = next((p for p in excel_data if str(p['Anonymized_Patient']) == str(patient_id)), None)
    if excel_patient:
        
        age_old = excel_patient['age_old']
        main_part = excel_patient['main_part']
        t1_t2_info = extract_t1_t2_info(main_part)
        return {
            'Anonymized_Patient': patient_id,
            'age_old': age_old,
            'T1_T2_Info': t1_t2_info
        }
    return None

async def download_from_pacs(server_url, username, password, output_dir, excel_file, num_samples=10):
    os.makedirs(output_dir, exist_ok=True)
    auth = aiohttp.BasicAuth(username, password)
    excel_data = read_patient_ids_from_excel(excel_file)

    async with aiohttp.ClientSession() as session:
        patients = await fetch_json(session, f"{server_url}/patients", auth)
        if not patients:
            return

        print(f"Found {len(patients)} patients on the server.")


        selected_patients = random.sample(patients, min(num_samples, len(patients)))

        tasks = [
            process_patient(session, server_url, patient, output_dir, auth, excel_data)
            for patient in selected_patients
        ]
        results = await asyncio.gather(*tasks)

        metadata = [result for result in results if result is not None]

        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False, encoding='utf-8')

if __name__ == "__main__":
    SERVER_URL = ""
    USERNAME = ""
    PASSWORD = ""
    EXCEL_FILE = ""
    OUTPUT_DIR = "./downloaded_pacs_data"
    asyncio.run(download_from_pacs(SERVER_URL, USERNAME, PASSWORD, OUTPUT_DIR, EXCEL_FILE))
