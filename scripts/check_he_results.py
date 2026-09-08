"""Reject a required integration suite that skipped its checks."""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("report", type=Path)
args = parser.parse_args()
root = ET.parse(args.report).getroot()
cases = root.findall(".//testcase")
if not cases or any(case.find("skipped") is not None for case in cases):
    raise SystemExit("Required integration tests must execute without dependency skips")
if any(case.find("failure") is not None or case.find("error") is not None for case in cases):
    raise SystemExit("Required integration validation failed")
print(f"{len(cases)} required integration tests executed without skips")
