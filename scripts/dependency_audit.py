#!/usr/bin/env python
"""Generate a reconciliation report between pyproject.toml constraints and requirements.txt locked versions.
Outputs:
  - Table of packages in both with version drift (caret/min vs exact)
  - Packages only in requirements.txt (transitives promoted) – optionally suggest adding if used directly.
  - Packages only in pyproject (likely not exported or optional extras removed)
Exit code 1 if critical drift (major version mismatch) detected.
"""
from __future__ import annotations
import re, sys, json
from pathlib import Path
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 fallback
from packaging.version import Version, InvalidVersion

PYPROJECT = Path('pyproject.toml')
REQ = Path('requirements.txt')

if not (PYPROJECT.exists() and REQ.exists()):
    print('pyproject.toml or requirements.txt missing', file=sys.stderr)
    sys.exit(2)

data = tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))
raw_deps = data.get('tool', {}).get('poetry', {}).get('dependencies', {})

# normalize poetry deps -> {name: spec}
poetry_deps = {}
for name, spec in raw_deps.items():
    if name == 'python':
        continue
    if isinstance(spec, str):
        poetry_deps[name.lower()] = spec
    elif isinstance(spec, dict):
        poetry_deps[name.lower()] = spec.get('version', '*')
    else:
        poetry_deps[name.lower()] = str(spec)

# parse requirements locked versions
req_versions = {}
lock_pattern = re.compile(r'^(?P<name>[A-Za-z0-9_.-]+)==(?P<ver>[^=]+)$')
for line in REQ.read_text(encoding='utf-8').splitlines():
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    m = lock_pattern.match(line)
    if not m:
        continue
    req_versions[m.group('name').lower()] = m.group('ver')

rows = []
critical = False
for name, spec in poetry_deps.items():
    locked = req_versions.get(name)
    drift = None
    severity = 'ok'
    base_spec = spec.replace('^','').replace('>=','').split(',')[0].strip()
    try:
        base_v = Version(re.sub(r'[^0-9A-Za-z.+-]','', base_spec)) if base_spec and base_spec not in ('*','') else None
    except InvalidVersion:
        base_v = None
    lock_v = None
    if locked:
        try:
            lock_v = Version(locked)
        except InvalidVersion:
            pass
    if base_v and lock_v:
        if base_v.major != lock_v.major:
            severity = 'major-drift'
            critical = True
        elif base_v.minor != lock_v.minor:
            severity = 'minor-drift'
        elif base_v.micro != lock_v.micro:
            severity = 'patch-drift'
        drift = f"{base_v} -> {lock_v}" if base_v != lock_v else ''
    rows.append({
        'package': name,
        'poetry_spec': spec,
        'locked': locked or '',
        'severity': severity,
        'drift': drift or ''
    })

only_locked = sorted(set(req_versions) - set(poetry_deps))
only_poetry = sorted(set(poetry_deps) - set(req_versions))

print('Dependency Reconciliation Report')
print('================================')
print('\nCore packages (in pyproject):')
for r in sorted(rows, key=lambda x: (x['severity'], x['package'])):
    print(f"- {r['package']}: poetry={r['poetry_spec']} locked={r['locked']} {('['+r['severity']+']') if r['severity']!='ok' else ''} {('drift:'+r['drift']) if r['drift'] else ''}")
if only_locked:
    print('Packages only in requirements (potential transitive / direct usage?):')
    for n in only_locked:
        print(f"  - {n}=={req_versions[n]}")
if only_poetry:
    print('Packages only in pyproject (maybe optional / not installed via export?):')
    for n in only_poetry:
        print(f"  - {n}:{poetry_deps[n]}")

summary = {
    'counts': {
        'core': len(rows),
        'only_locked': len(only_locked),
        'only_poetry': len(only_poetry)
    },
    'drift_breakdown': {
        'major': sum(1 for r in rows if r['severity']=='major-drift'),
        'minor': sum(1 for r in rows if r['severity']=='minor-drift'),
        'patch': sum(1 for r in rows if r['severity']=='patch-drift'),
    }
}
print('\nJSON Summary:')
print(json.dumps(summary, indent=2))

if critical:
    sys.exit(1)
