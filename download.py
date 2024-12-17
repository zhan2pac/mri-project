import asyncio
import aiohttp
import os
import pandas as pd
import random

def read_patient_ids_from_excel(file_path):
    df = pd.read_excel(file_path)
    return df['Anonymized_Patient'].dropna().tolist()

async def fetch_json(session, url, auth):
    async with session.get(url, auth=auth) as response:
        if response.status != 200:
            print(f"Failed to fetch {url}: {response.status} {await response.text()}")
            return None
        return await response.json()

async def download_file(session, url, file_path, auth):
    async with session.get(url, auth=auth) as response:
        if response.status != 200:
            print(f"Failed to download file from {url}: {response.status} {await response.text()}")
            return
        with open(file_path, "wb") as f:
            f.write(await response.read())

async def process_patient(session, server_url, patient, output_dir, auth):
    patient_data = await fetch_json(session, f"{server_url}/patients/{patient}", auth)
    if not patient_data:
        return

    patient_id = patient_data['MainDicomTags'].get('PatientID')

    print(f"Downloading data for patient: {patient_id}")
    patient_dir = os.path.join(output_dir, patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    studies = await fetch_json(session, f"{server_url}/patients/{patient}/studies", auth)
    if not studies:
        return

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

                print(f"Downloading instance {instance_id['ID']}")
                await download_file(session, f"{server_url}/instances/{instance_id['ID']}/file", dicom_path, auth)
                await download_file(session, f"{server_url}/instances/{instance_id['ID']}/preview", dicom_preview_path, auth)

async def download_from_pacs(server_url, username, password, output_dir, num_samples=10):
    os.makedirs(output_dir, exist_ok=True)
    auth = aiohttp.BasicAuth(username, password)

    async with aiohttp.ClientSession() as session:
        patients = await fetch_json(session, f"{server_url}/patients", auth)
        if not patients:
            return

        print(f"Found {len(patients)} patients on the server.")

        # Randomly select 10 patients
        selected_patients = random.sample(patients, min(num_samples, len(patients)))

        tasks = [
            process_patient(session, server_url, patient, output_dir, auth)
            for patient in selected_patients
        ]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    SERVER_URL = ""
    USERNAME = ""
    PASSWORD = ""
    OUTPUT_DIR = "./downloaded_pacs_data"
    asyncio.run(download_from_pacs(SERVER_URL, USERNAME, PASSWORD, OUTPUT_DIR))
