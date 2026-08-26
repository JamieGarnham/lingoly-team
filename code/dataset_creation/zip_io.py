"""Shared helpers for reading/writing the password-protected .jsonl.zip
benchmark splits in data/splits/. The zip password doubles as a light
anti-scraping gate on the dataset, matching code/benchmarking/*.py.
"""

import json

import pyzipper

ZIP_PW = "olympiad"


def read_jsonl_zip(path):
    """Yield parsed JSON objects from a password-protected .jsonl.zip file."""
    with pyzipper.AESZipFile(path) as zf:
        name = zf.namelist()[0]
        data = zf.read(name, pwd=ZIP_PW.encode())
    for line in data.decode("utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def write_jsonl_zip(path, entries):
    """Write an iterable of JSON-serializable objects to a password-protected .jsonl.zip file."""
    jsonl_bytes = "".join(
        json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries
    ).encode("utf-8")
    arcname = path.split("/")[-1].removesuffix(".zip")
    with pyzipper.AESZipFile(path, "w", encryption=pyzipper.WZ_AES) as zf:
        zf.setpassword(ZIP_PW.encode())
        zf.writestr(arcname, jsonl_bytes)
