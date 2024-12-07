import sys
from scholarly import scholarly
import bibtexparser
from scholarly import ProxyGenerator

import re
import requests
from bs4 import BeautifulSoup


def extract_bibtex_link(url_scholarbib):
    # Make a request to the URL
    response = requests.get(url_scholarbib)

    # Check if the response returns successfully with status code 200
    if response.status_code != 200:
        # raise Exception(f"Failed to fetch page {url_scholarbib}: status code {response.status_code}")
        print(f"Failed to fetch page {url_scholarbib}: status code {response.status_code}")

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'lxml')

    # Find the link containing the text 'BibTeX'
    bibtex_link = soup.find('a', string=re.compile('BibTeX'))

    # Check if the link is found
    if bibtex_link and bibtex_link.has_attr('href'):
        # Extract the URL
        bibtex_url = bibtex_link['href']
        
        # Make a request to the URL and get the content
        bib_response = requests.get(bibtex_url)
        
        # Check if the request was successful
        if bib_response.status_code != 200:
            print(f"Failed to download the content, status code: {bib_response.status_code}")
    else:
        print("BibTeX link not found in the provided HTML content.")
    return bib_response.text


def find_all_versions(pub):
    # Find all versions of this paper from Google Scholar

    # ('citedby_url', '/scholar?cites=9259844133671260448&as_sdt=2005&sciodt=0,5&hl=en')
    # extract the cluster ID of the first version
    cluster_id = re.search(r'cites=([^&]+)', pub['citedby_url']).group(1)
    url = f'https://scholar.google.com/scholar?cluster={cluster_id}&hl=en&as_sdt=0,20'
    print(f"Fetching all versions from {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    raw_versions = []
    for x in soup.find_all('div', {'class': 'gs_r gs_or gs_scl'}):
        raw_versions.append(x['data-cid'])

    print(f"Found {len(raw_versions)} versions")

    
    # For each of these versions, extract the url_scholarbib
    # /scholar?hl=en&q=info:INH1xi2TgYAJ:scholar.google.com/&output=cite&scirp=0&hl=en
    entries = []
    for i, version in enumerate(raw_versions):
        # print(f"Fetching bibtex for version {i+1}/{len(raw_versions)}")
        url_scholarbib = f'https://scholar.google.com/scholar?q=info:{version}:scholar.google.com/&output=cite&scirp=0&hl=en'
        bibtex = extract_bibtex_link(url_scholarbib)
        # parse bibtext to replace the entry
        entry = bibtexparser.loads(bibtex).entries[0]
        entries.append(entry)

    import ipdb; ipdb.set_trace()

    return entries



# Function to parse .bib file and get entries
def parse_bib_file(file_path):
    with open(file_path) as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)
    return bib_database.entries

# Function to search for publication venues using scholarly
def search_publication_venues(query):
    pub = scholarly.search_single_pub(query)
    # Try to get the first publication that matches the query
    url_scholarbib = "https://scholar.google.com" + pub["url_scholarbib"]
    bibtex = extract_bibtex_link(url_scholarbib)
    # parse bibtext to replace the entry
    replacement_entry = bibtexparser.loads(bibtex).entries[0]
    return replacement_entry

# Function to update the .bib file with the chosen venue
def update_bib_file(file_path, entries):
    db = bibtexparser.bibdatabase.BibDatabase()
    db.entries = entries
    with open(file_path, 'w') as bibtex_file:
        bibtexparser.dump(db, bibtex_file)

def find_first_version(query):
    pub = scholarly.search_single_pub(query)
    # Try to get the first publication that matches the query
    replacement_entry = scholarly.bibtex(pub)
    return [replacement_entry]

# Main function to orchestrate the script
def main(bib_file_path):

    # Set up a ProxyGenerator object to use free proxies
    # This needs to be done only once per session
    pg = ProxyGenerator()
    pg.FreeProxies()
    scholarly.use_proxy(pg)
    entries = parse_bib_file(bib_file_path)
    updated_entries = []
    for entry in entries:
        # Only parse arxiv entries
        if not re.match(r'.*arXiv.*', str(entry)):
            # print(f"Skipping entry: {entry}\n\n")
            updated_entries.append(entry)
            continue

        title = entry.get('title')
        print(f"{title}")
        # new_entry = find_first_version(title)
        # print(f"Found venue: {new_entry}\n\n")
        # Prompt the user to accept the found venue or skip
        # user_input = input("Accept this venue? (Y/n): ")
        # if user_input.lower() == 'y':
            # updated_entries.append(new_entry)
            
    # Update the .bib file with all the changes
    # update_bib_file(f"updated_{bib_file_path}", entries)

# Run the script
bib_file_path = sys.argv[1]
main(bib_file_path)
