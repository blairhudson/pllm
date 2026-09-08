"""Run each required integration test in a fresh process and reject skips."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--marker', choices=['he', 'sdk'], required=True)
    parser.add_argument('--output', type=Path, default=Path('results/integration'))
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    collected = subprocess.run(
        [sys.executable, '-m', 'pytest', '--collect-only', '-q', '-m', args.marker],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    nodes = [line for line in collected.stdout.splitlines()
             if line.startswith('tests/') and '::' in line]
    if not nodes:
        raise SystemExit('No tests were collected for the required lane')
    combined = ET.Element('testsuites')
    records = []
    failed = False
    for i, node in enumerate(nodes):
        report = output / f'{i:03}.xml'
        log = output / f'{i:03}.log'
        started = time.perf_counter()
        with log.open('w') as stream:
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pytest', '-q', node, f'--junitxml={report}'],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout,
                )
                status = result.returncode
            except subprocess.TimeoutExpired:
                status = 'timeout'
        if report.exists():
            report_root = ET.parse(report).getroot()
            suites = [report_root] if report_root.tag == 'testsuite' else list(report_root)
            cases = report_root.findall('.//testcase')
            valid = bool(cases) and not any(
                any(case.find(tag) is not None for tag in ('skipped', 'error', 'failure'))
                for case in cases
            )
            combined.extend(suites)
        else:
            valid = False
            suite = ET.SubElement(combined, 'testsuite', name=node, tests='1', errors='1')
            case = ET.SubElement(suite, 'testcase', name=node)
            ET.SubElement(case, 'error', message=f'Process exit: {status}')
        passed = status == 0 and valid
        failed |= not passed
        records.append({'test': node, 'passed': passed, 'exit': status,
                        'seconds': time.perf_counter() - started, 'log': log.name})
        print(f'{"PASS" if passed else "FAIL"} {node}', flush=True)
    ET.ElementTree(combined).write(output / 'junit.xml', encoding='utf-8', xml_declaration=True)
    (output / 'summary.json').write_text(json.dumps(records, indent=2) + '\n')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
