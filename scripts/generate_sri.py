"""
generate_sri.py -- Compute the SRI integrity hash for the Plotly CDN script.

Usage:
    python scripts/generate_sri.py

Output:
    Prints the integrity attribute value to stdout, e.g.:
        integrity="sha384-<hash>"

    Copy that value into dashboard/index.html on the Plotly <script> tag:
        <script src="..." crossorigin="anonymous" integrity="sha384-<hash>"></script>
"""

import hashlib
import base64
import urllib.request
import logging

log = logging.getLogger(__name__)

URL = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"


def compute_sri(url: str) -> str:
    log.info(f"Fetching {url} ...")
    with urllib.request.urlopen(url) as response:
        data = response.read()
    digest = hashlib.sha384(data).digest()
    hash_b64 = base64.b64encode(digest).decode()
    return f'integrity="sha384-{hash_b64}"'


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = compute_sri(URL)
    log.info(result)
    print()
    log.info("Add that attribute to the Plotly <script> tag in dashboard/index.html:")
    log.info(f'  <script src="{URL}" crossorigin="anonymous" {result}></script>')
