"""Downloads the files the code reads at run time into the data directory, and checks each one.

  uv run ballet-fetch-data                        # everything main needs: zenodo and pools
  uv run ballet-fetch-data zenodo --start-states 4r_10t_30o   # one start-state file, not all four
  uv run ballet-fetch-data --verify               # check what is there, download nothing

zenodo: the original's data/mujoco_world, data/paper_test_tasksets and data/start_states, from the
  one archive of Zenodo record 16546403 (roboballet.tar.zst, 1.1 GB). A .tar.zst cannot be read in
  parts, so the whole archive streams through and only these files are written, 1.7 GB with all
  four start-state files. --archive reads a copy already on disk instead.
pools: the large scene pools, from this repository's release assets. The small ones are in git.
test-pools: the mesh_tool pools, which only the test branch's E10 test reads.

Every file is checked against data/manifest.json, its size and sha256, and the archive against
Zenodo's md5. A file that is already there and matches is not downloaded again.

The repository is private, so its release assets need credentials. With the GitHub CLI logged in,
`gh release download` fetches them. Otherwise GH_TOKEN or GITHUB_TOKEN is sent to the GitHub API,
and never printed. The repository is --repo, else $ROBOBALLET_RELEASE_REPO, else the clone's origin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import BinaryIO
import urllib.request

from roboballet_mjlab.paths import DATA_ENV, PACKAGE_DATA, PACKAGE_DIR, data_dir

MANIFEST = PACKAGE_DATA / 'manifest.json'
ZENODO_DIRS = ('mujoco_world', 'paper_test_tasksets', 'start_states')
CHUNK = 1 << 20
WHAT = ('zenodo', 'pools', 'test-pools')


def sha256_of(path: Path) -> str:
  digest = hashlib.sha256()
  with open(path, 'rb') as f:
    while block := f.read(CHUNK):
      digest.update(block)
  return digest.hexdigest()


def matches(path: Path, size: int, sha256: str) -> bool:
  return path.is_file() and path.stat().st_size == size and sha256_of(path) == sha256


class HashingReader:
  """Reads through to `raw`, hashing every byte and counting them, for the archive's md5."""

  def __init__(self, raw: BinaryIO, total: int) -> None:
    self.raw, self.total, self.done = raw, total, 0
    self.md5 = hashlib.md5()
    self._shown = -1

  def read(self, size: int = -1) -> bytes:
    block = self.raw.read(size)
    self.md5.update(block)
    self.done += len(block)
    percent = self.done * 100 // max(self.total, 1)
    if percent != self._shown and percent % 5 == 0:
      self._shown = percent
      print(f'  archive {self.done / 1e9:.2f} of {self.total / 1e9:.2f} GB', flush=True)
    return block

  def drain(self) -> None:
    while self.read(CHUNK):
      pass


def write_checked(source: BinaryIO, dest: Path, size: int | None, sha256: str | None) -> None:
  """To a temporary file beside `dest`, renamed into place only if size and sha256 match."""
  dest.parent.mkdir(parents=True, exist_ok=True)
  tmp = dest.with_name(f'{dest.name}.part{os.getpid()}')
  digest, written = hashlib.sha256(), 0
  try:
    with open(tmp, 'wb') as f:
      while block := source.read(CHUNK):
        digest.update(block)
        written += len(block)
        f.write(block)
    if size is not None and (written != size or digest.hexdigest() != sha256):
      raise RuntimeError(f'{dest}: {written} bytes, sha256 {digest.hexdigest()}; expected {size} bytes, sha256 {sha256}')
    os.replace(tmp, dest)
  finally:
    tmp.unlink(missing_ok=True)


def zenodo_wanted(manifest: dict, start_states: list[str] | None) -> dict[str, list]:
  files = manifest['zenodo']['files']
  wanted = {}
  for rel, entry in files.items():
    top, name = rel.split('/', 1)[0], PurePosixPath(rel).name
    if top == 'start_states' and start_states is not None and name.removesuffix('.pb') not in start_states:
      continue
    wanted[rel] = entry
  return wanted


def fetch_zenodo(manifest: dict, root: Path, start_states: list[str] | None, archive: Path | None) -> None:
  import zstandard

  zenodo = manifest['zenodo']
  wanted = zenodo_wanted(manifest, start_states)
  missing = {rel: entry for rel, entry in wanted.items() if not matches(root / rel, *entry)}
  print(f'zenodo: {len(wanted) - len(missing)} of {len(wanted)} files already in {root}')
  if not missing:
    return
  prefix = zenodo['prefix']
  if archive is not None:
    raw, total = open(archive, 'rb'), archive.stat().st_size
    print(f'reading {archive}')
  else:
    raw = urllib.request.urlopen(zenodo['url'], timeout=60)
    total = zenodo['size']
    print(f'downloading {zenodo["archive"]} of Zenodo record {zenodo["record"]}, {total / 1e9:.2f} GB')
  found = set()
  with raw:
    reader = HashingReader(raw, total)
    stream = zstandard.ZstdDecompressor().stream_reader(reader, read_across_frames=True)
    with tarfile.open(fileobj=stream, mode='r|') as tar:
      for member in tar:
        rel = member.name.removeprefix('./')
        if not rel.startswith(prefix):
          continue
        rel = rel[len(prefix):]
        parts = PurePosixPath(rel).parts
        if rel not in missing or not member.isfile() or '..' in parts or PurePosixPath(rel).is_absolute():
          continue
        write_checked(tar.extractfile(member), root / rel, *missing[rel])
        found.add(rel)
        print(f'  {rel}: {missing[rel][0]:,} bytes, sha256 matches')
    reader.drain()
  if reader.done != zenodo['size'] or reader.md5.hexdigest() != zenodo['md5']:
    raise RuntimeError(f'archive: {reader.done} bytes, md5 {reader.md5.hexdigest()}; Zenodo lists {zenodo["size"]}, '
                       f'md5 {zenodo["md5"]}')
  print(f'  archive md5 matches Zenodo, {zenodo["md5"]}')
  absent = sorted(set(missing) - found)
  if absent:
    raise RuntimeError(f'not in the archive: {absent}')


def origin_repo() -> str | None:
  """OWNER/NAME of the clone's GitHub origin, read from .git/config without running git."""
  config = PACKAGE_DIR.parents[1] / '.git' / 'config'
  if not config.is_file():
    return None
  match = re.search(r'\[remote "origin"\][^\[]*?url\s*=\s*(\S+)', config.read_text())
  if match is None:
    return None
  found = re.search(r'github\.com[:/]([^/\s]+/[^/\s]+?)(?:\.git)?$', match.group(1))
  return found.group(1) if found else None


def download_asset(repo: str, tag: str, name: str, dest: Path, size: int, sha256: str) -> None:
  if shutil.which('gh'):
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
      done = subprocess.run(['gh', 'release', 'download', tag, '--repo', repo, '--pattern', name, '--dir', tmp])
      if done.returncode == 0:
        with open(Path(tmp) / name, 'rb') as f:
          write_checked(f, dest, size, sha256)
        return
      print(f'  gh release download failed with status {done.returncode}, trying the API', flush=True)
  token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
  if token:
    api = f'https://api.github.com/repos/{repo}/releases/tags/{tag}'
    request = urllib.request.Request(api, headers={'Accept': 'application/vnd.github+json'})
    request.add_unredirected_header('Authorization', f'Bearer {token}')
    with urllib.request.urlopen(request, timeout=60) as response:
      assets = {asset['name']: asset['url'] for asset in json.load(response)['assets']}
    if name not in assets:
      raise RuntimeError(f'release {tag} of {repo} has no asset {name}')
    # Unredirected, so the token does not follow the redirect to the storage host.
    request = urllib.request.Request(assets[name], headers={'Accept': 'application/octet-stream'})
    request.add_unredirected_header('Authorization', f'Bearer {token}')
  else:
    request = urllib.request.Request(f'https://github.com/{repo}/releases/download/{tag}/{name}')
  with urllib.request.urlopen(request, timeout=60) as response:
    write_checked(response, dest, size, sha256)


def fetch_pools(manifest: dict, root: Path, group: str, repo: str | None, tag: str | None) -> None:
  release = manifest['release']
  tag = tag or release['tag']
  assets = {name: entry for name, entry in release['assets'].items() if entry['group'] == group}
  todo = {name: entry for name, entry in assets.items() if not matches(root / entry['path'], entry['size'], entry['sha256'])}
  print(f'{"pools" if group == "main" else "test-pools"}: {len(assets) - len(todo)} of {len(assets)} files already in {root}')
  if not todo:
    return
  repo = repo or os.environ.get('ROBOBALLET_RELEASE_REPO') or origin_repo()
  if repo is None:
    raise SystemExit('Which GitHub repository holds the release? Give --repo OWNER/roboballet-mjlab.')
  for name, entry in todo.items():
    print(f'downloading {name} from release {tag} of {repo}, {entry["size"] / 1e6:.0f} MB', flush=True)
    dest = root / entry['path']
    dest.parent.mkdir(parents=True, exist_ok=True)
    download_asset(repo, tag, name, dest, entry['size'], entry['sha256'])
    print(f'  {entry["path"]}: {entry["size"]:,} bytes, sha256 matches')


def verify(manifest: dict, root: Path, what: list[str], start_states: list[str] | None) -> bool:
  entries = []
  if 'zenodo' in what:
    entries += [(rel, size, sha) for rel, (size, sha) in zenodo_wanted(manifest, start_states).items()]
  for group, label in (('main', 'pools'), ('test', 'test-pools')):
    if label in what:
      entries += [(e['path'], e['size'], e['sha256']) for e in manifest['release']['assets'].values() if e['group'] == group]
  bad = [rel for rel, size, sha in entries if not matches(root / rel, size, sha)]
  for rel in bad:
    print(f'missing or different: {root / rel}')
  print(f'{len(entries) - len(bad)} of {len(entries)} files in {root} match the manifest')
  return not bad


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(prog='ballet-fetch-data', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('what', nargs='*', choices=WHAT, help='Default: zenodo and pools.')
  parser.add_argument('--data-dir', type=Path, help=f'Default: ${DATA_ENV}, else data/ in the clone.')
  parser.add_argument('--start-states', nargs='+', metavar='NAME',
                      help='Only these start-state files, by name without .pb, e.g. 4r_10t_30o. Default: all four.')
  parser.add_argument('--archive', type=Path, help='A roboballet.tar.zst already on disk.')
  parser.add_argument('--repo', help='OWNER/NAME of the GitHub repository whose release holds the pools.')
  parser.add_argument('--tag', help='Release tag. Default: the manifest\'s.')
  parser.add_argument('--verify', action='store_true', help='Check the files against the manifest, download nothing.')
  args = parser.parse_args(argv)
  if args.data_dir is not None:
    os.environ[DATA_ENV] = str(args.data_dir.expanduser().resolve())
  what = args.what or ['zenodo', 'pools']
  root = data_dir()
  manifest = json.loads(MANIFEST.read_text())
  if args.verify:
    return 0 if verify(manifest, root, what, args.start_states) else 1
  root.mkdir(parents=True, exist_ok=True)
  if 'zenodo' in what:
    fetch_zenodo(manifest, root, args.start_states, args.archive)
  if 'pools' in what:
    fetch_pools(manifest, root, 'main', args.repo, args.tag)
  if 'test-pools' in what:
    fetch_pools(manifest, root, 'test', args.repo, args.tag)
  return 0


if __name__ == '__main__':
  sys.exit(main())
