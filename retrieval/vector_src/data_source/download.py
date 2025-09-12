import os
import wget

file_links = [
    {
        "title": "Lung cancer symptoms at diagnosis: results of a nationwide registry study",
        "url": "https://www.esmoopen.com/action/showPdf?pii=S2059-7029%2820%2932756-3"
    },
    {
        "title": """Symptoms and signs of lung cancer prior to diagnosis: case–control study using electronic 
        health records from ambulatory care within a large USbased tertiary care centre""",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10124310/pdf/bmjopen-2022-068832.pdf"
    },
    {
        "title": "Automatic Cough Analysis for Non-Small Cell Lung Cancer Detection",
        "url": "https://arxiv.org/pdf/2507.19174"
    },
    {
        "title": """Handling uncertainty using features from pathology: opportunities in primary care data 
        for developing high risk cancer survival methods""",
        "url": "https://arxiv.org/pdf/2012.09976"
    }
]


def is_exist(file_link):
    filename = f"retrieval/vector_src/data_source/{file_link['title']}.pdf"
    return os.path.exists(filename)


for file_link in file_links:
    if not is_exist(file_link):
        print(f"Downloading: {file_link['title']}")
        wget.download(
            file_link["url"], out=f"retrieval/vector_src/data_source/{file_link['title']}.pdf")
