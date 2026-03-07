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

URL = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"


def compute_sri(url: str) -> str:
    print(f"Fetching {url} ...")
    with urllib.request.urlopen(url) as response:
        data = response.read()
    digest = hashlib.sha384(data).digest()
    hash_b64 = base64.b64encode(digest).decode()
    return f'integrity="sha384-{hash_b64}"'


if __name__ == "__main__":
    result = compute_sri(URL)
    print(result)
    print()
    print("Add that attribute to the Plotly <script> tag in dashboard/index.html:")
    print(f'  <script src="{URL}" crossorigin="anonymous" {result}></script>')
