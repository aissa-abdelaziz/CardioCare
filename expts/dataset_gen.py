import os
import json
from PyPDF2 import PdfReader

def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        pdf_reader = PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def create_dataset(pdf_files, output_json):
    dataset = []

    for pdf_file in pdf_files:
        text = read_pdf(pdf_file)

        dataset.append({"text": text.strip()})  

    with open(output_json, "w") as json_file:
        json.dump(dataset, json_file, ensure_ascii=False, indent=4)

    print(f"Dataset created and saved to {output_json}")

def main():
    pdf_directory = "~/data" 
    output_json = "fine_tuning_dataset.json"      

    pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

    create_dataset(pdf_files, output_json)

if __name__ == "__main__":
    main()
