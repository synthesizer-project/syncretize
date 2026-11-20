import gzip
import os
import shutil
import sys
import tarfile
from urllib.parse import parse_qs, quote, urlparse

import numpy as np
import requests
from tqdm import tqdm

# import function from incident_utils module
# allow it to see anything two folders up so that grid_utils and
# incident_utils can be accessed
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
# import incident_utils

# __tag__ = incident_utils.__tag__


def get_model_filename(model, axis_name=None):
    """Build an SPS grid filename following Synthesizer conventions."""
    synthesizer_model_name = f"{model['sps_name']}"

    if model["sps_version"] is not False:
        synthesizer_model_name += f"-{model['sps_version']}"

    if model["sps_variant"] is not False:
        synthesizer_model_name += f"-{model['sps_variant']}"

    # Mass limits label
    mass_limits_label = ",".join(
        map(lambda x: str(np.round(x, 2)), model["imf_masses"])
    )

    synthesizer_model_name += f"_{model['imf_type']}-{mass_limits_label}"

    # IMF slope label
    if model["imf_type"] == "bpl":
        imf_slopes_label = ",".join(
            map(lambda x: str(np.round(x, 2)), model["imf_slopes"])
        )
        synthesizer_model_name += "-" + imf_slopes_label

    # If alpha enhancement varies then say so
    if model["alpha_enhancement"] is not False:
        synthesizer_model_name += f"_alpha{model['alpha_enhancement']}"

    # Higher-dimensional axes extend the canonical SPS/IMF name.
    if axis_name is not None:
        synthesizer_model_name += f"_axis-{axis_name}"

    return synthesizer_model_name


def download_sharepoint_file(share_url, path, destination):
    """
    Download a file from a public SharePoint folder link.

    Opening the folder's sharing link gives the session access to the folder
    (no login), and SharePoint's REST API then serves its files.

    Args:
        share_url (str)
            The folder's sharing link, e.g.
            "https://<tenant>.sharepoint.com/:f:/s/<site>/<token>".
        path (str)
            The file's path within the folder, e.g. "subfolder/file.tar.gz".
        destination (str)
            The file to write.
    """
    session = requests.Session()
    response = session.get(share_url, timeout=60)
    response.raise_for_status()

    # The link redirects to the folder's page, whose "id" is the folder's
    # path on the server, /sites/<site>/<library>/...
    page = urlparse(response.url)
    folder = parse_qs(page.query)["id"][0]
    site = f"{page.scheme}://{page.netloc}" + "/".join(folder.split("/")[:3])
    url = (
        f"{site}/_api/web/GetFileByServerRelativeUrl("
        f"'{quote(f'{folder}/{path}')}')/$value"
    )

    # Written under a temporary name, so an interrupted download isn't
    # mistaken for a whole one
    with session.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(destination + ".part", "wb") as file, tqdm(
            total=total, unit="B", unit_scale=True, desc=os.path.basename(path)
        ) as progress:
            for chunk in response.iter_content(1 << 20):
                file.write(chunk)
                progress.update(len(chunk))
    os.replace(destination + ".part", destination)


def extract_gzipped_tables(archive, directory, prefixes):
    """
    Extract the gzipped tables (<name>.gz) an archive holds, decompressed.

    Args:
        archive (str)
            The (tar) archive.
        directory (str)
            The directory to write the tables to (as <name>).
        prefixes (tuple of str)
            Only tables whose names start with one of these are extracted.

    Returns:
        list of str
            The tables written.
    """
    os.makedirs(directory, exist_ok=True)
    written = []
    with tarfile.open(archive) as tar:
        for member in tar:
            name = os.path.basename(member.name)
            if not member.isfile() or not name.startswith(tuple(prefixes)):
                continue
            path = os.path.join(directory, name.removesuffix(".gz"))
            with tar.extractfile(member) as packed, open(path, "wb") as out:
                if name.endswith(".gz"):
                    with gzip.open(packed) as table:
                        shutil.copyfileobj(table, out)
                else:
                    shutil.copyfileobj(packed, out)
            written.append(path)
    return written
