import subprocess
from urllib.request import urlretrieve
import requests
import json
from collections import Counter

import pandas as pd


def main():
    minimum_ensemble_members = 10

    # ua, va, psl
    variable = "ua"
    # day, Amon
    frequency = "day"

    generate_csv(variable, frequency, minimum_ensemble_members)
    run_wget_script(variable, frequency)


def generate_csv(variable, frequency, minimum_ensemble_members):
    """Generate a CSV file with the data listing for all model data in the CMIP6
    historical simulations that match the variable and frequency

    This is done by generating the URL to search ESGF but returning all results in JSON
    format. Then taking this JSON file and filtering it for the data needed

    Args:
        variable (str): The CMIP6 variable name (variable_id)
        frequency (str): The CMIP6 frequency of output (table_id)
        minimum_ensemble_members (int): Filter out any models with less than this number
            of ensemble members
    """
    url = (
        "https://esgf.ceda.ac.uk/esg-search/search/"
        "?limit=1000"
        "&type=Dataset"
        "&replica=false"
        "&latest=true"
        "&experiment_id=historical"
        f"&table_id={frequency}"
        f"&variable_id={variable}"
        "&format=application%2Fsolr%2Bjson"
    )

    entries = get_json_summary(url)
    with open(f"result_{variable}_{frequency}.json", "w") as f:
        json.dump(entries, f)

    with open(f"result_{variable}_{frequency}.json", "r") as f:
        entries = json.load(f)

    # Match full historical period 1850-2014
    to_remove = []
    for entry in entries:
        try:
            start = entry["datetime_start"]
            if "1850" not in start and "1849" not in start:
                to_remove.append(entry)
        except KeyError:
            to_remove.append(entry)
    for entry in to_remove:
        entries.remove(entry)
    print(f"Removed {len(to_remove)} entries not starting in 1850")

    to_remove = []
    for entry in entries:
        try:
            end = entry["datetime_stop"]
        except KeyError:
            end = entry["datetime_end"]
        if "2014" not in end and "2015" not in end:
            to_remove.append(entry)
    for entry in to_remove:
        entries.remove(entry)
    print(f"Removed {len(to_remove)} entries not ending in 2014")

    # Remove duplicates
    models = [(entry["source_id"][0], entry["variant_label"][0]) for entry in entries]
    duplicates = [entries[n] for n, model in enumerate(models) if model in models[:n] + models[n + 1:]]
    for entry in duplicates[::2]:
        entries.remove(entry)
    print(f"Removed {len(duplicates) // 2} duplicate entries")

    # Filter out models with too few ensemble members
    models = [entry["source_id"][0] for entry in entries]
    number_of_runs_by_model = Counter(models)
    to_remove = []
    for entry in entries:
        if number_of_runs_by_model[entry["source_id"][0]] < minimum_ensemble_members:
            to_remove.append(entry)
    for entry in to_remove:
        entries.remove(entry)
    print(f"Removed {len(to_remove)} entries with less than {minimum_ensemble_members} "
          f"ensemble members")

    models = [entry["source_id"][0] for entry in entries]
    number_of_runs_by_model = Counter(models)
    print(number_of_runs_by_model)

    df = pd.DataFrame(entries)

    # Fix lists of length 1
    for col in df:
        if df[col].apply(lambda x: type(x) == list and len(x) == 1).all():
            df[col] = df[col].apply(lambda x: x[0])

    df.to_csv(f"result_{variable}_{frequency}.csv")


def get_json_summary(url):
    """Run the query for ESGF given by the url and then open the resulting JSON file as
    a dictionary
    """
    r = requests.get(url).json()
    entries = r["response"]["docs"]
    entries = sorted(entries, key=lambda x: x["source_id"] + x["variant_label"])

    return entries


def extract_matching(variable, frequency, variable_to_match, frequency_to_match):
    df_input = pd.read_csv(f"result_{variable}_{frequency}.csv")
    df_to_match = pd.read_csv(f"result_{variable_to_match}_{frequency_to_match}.csv")

    df = pd.merge(
        df_input,
        df_to_match,
        on=["source_id", "variant_label"],
        how="inner",
        suffixes=(None, "_old")
    )
    df.drop([col for col in df if "_old" in col], axis=1, inplace=True)

    return df


def run_wget_script(variable, frequency, max_retries=10):
    """Go through a CSV from generate_csv and generate a wget script for each model
    listed and then run that wget script to download the data

    Args:
        variable (str): The CMIP6 variable name (variable_id)
        frequency (str): The CMIP6 frequency of output (table_id)
        max_retries (int): The wget script often misses some files and will need
            rerunning. We don't want to do this indefinitely in case the server is just
            down, so give up after this many retries and print a message to try again
            later
    """
    df = pd.read_csv(f"result_{variable}_{frequency}.csv")

    for model in sorted(list(set(df.source_id))):
        print(model)

        df_model = df[df.source_id == model]
        wget_script = f"wget_psl_{model}.sh"
        generate_download_script(df_model, script_filename=wget_script)

        # Run the wget script until all the files have downloaded
        # Check the wget script has downloaded all the files
        n_retries = 0
        while not all_files_downloaded(df_model, wget_script) and n_retries < max_retries:
            subprocess.run(["bash", wget_script, "-s"])
            n_retries += 1

        if n_retries == max_retries and not all_files_downloaded(df_model, wget_script):
            print(f"Not all files downloaded for {model}. Try rerunning later")


def all_files_downloaded(df, wget_script):
    try:
        with open(f".{wget_script}.status", "r") as f:
            txt = "".join(f.readlines())
    except FileNotFoundError:
        return False

    return df.variant_label.apply(lambda x: x in txt).all()

# This is what an example URL for generating a wget script looks like
# url = (
#     "https://esgf-node.llnl.gov/esg-search/wget/?distrib=false"
#     "&dataset_id=CMIP6.CMIP.NCAR.CESM2.historical.r10i1p1f1.day.va.gn.v20190313|esgf-data.ucar.edu"
#     "&dataset_id=CMIP6.CMIP.NCAR.CESM2.historical.r9i1p1f1.day.va.gn.v20190311|esgf-data.ucar.edu"
#     "&dataset_id=CMIP6.CMIP.NCAR.CESM2.historical.r7i1p1f1.day.va.gn.v20190311|esgf-data.ucar.edu"
#     "&dataset_id=CMIP6.CMIP.NCAR.CESM2.historical.r8i1p1f1.day.va.gn.v20190311|esgf-data.ucar.edu"
#     "&dataset_id=CMIP6.CMIP.NCAR.CESM2.historical.r4i1p1f1.day.va.gn.v20190308|esgf-data.ucar.edu"
#     "&dataset_id=CMIP6.CMIP.NCAR.CESM2.historical.r6i1p1f1.day.va.gn.v20190308|esgf-data.ucar.edu"
#     "&dataset_id=CMIP6.CMIP.NCAR.CESM2.historical.r5i1p1f1.day.va.gn.v20190308|esgf-data.ucar.edu"
#     "&dataset_id=CMIP6.CMIP.NCAR.CESM2.historical.r3i1p1f1.day.va.gn.v20190308|esgf-data.ucar.edu"
#     "&dataset_id=CMIP6.CMIP.NCAR.CESM2.historical.r2i1p1f1.day.va.gn.v20190308|esgf-data.ucar.edu"
#     "&dataset_id=CMIP6.CMIP.NCAR.CESM2.historical.r1i1p1f1.day.va.gn.v20190308|esgf-data.ucar.edu"
#     "&dataset_id=CMIP6.CMIP.NCAR.CESM2.historical.r11i1p1f1.day.va.gn.v20190514|esgf-data.ucar.edu"
# )


data_nodes = [
    "esgf-node.llnl.gov",
    "esgf-node.ipsl.upmc.fr",
    "esgf-data.dkrz.de",
    "esgf.ceda.ac.uk"
]


def generate_download_script(entries, script_filename="wget_cmip.sh"):
    """Generate a download script for a set of data for one model

    Most of the model data seems to be stored on one of the four nodes listed above but
    always on the same node for a given model. So try to generate the wget script by
    model for each node until it does correctly generate a script

    Args:
        entries (pandas.DataFrame): The set of individual data files associated with the
            model
        script_filename (str): Filename to save the script to
    """

    n = 0
    while not genuine_download_script(script_filename) and n < len(data_nodes):
        url = (
            f"https://{data_nodes[n]}/esg-search/wget/?distrib=false" +
            "&dataset_id=".join([""] + [dataset_id for dataset_id in entries.id])
        )

        print(url)
        urlretrieve(url, script_filename)

        n += 1

    if n == len(data_nodes):
        print(f"failed for {script_filename}")


def genuine_download_script(filename):
    """Check that the script has downloaded and is not just an empty file
    """
    try:
        with open(filename) as f:
            text = f.readlines()
    except FileNotFoundError:
        text = []

    return len(text) > 1


if __name__ == '__main__':
    main()
