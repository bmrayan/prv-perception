"""Check code syntax, local links, and the external saved-result hashes."""
import ast
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from reproduce import ROOT, REPO_ROOT

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.links += [attrs[k] for k in ('href', 'src') if k in attrs]
        if 'id' in attrs:
            self.ids.add(attrs['id'])

def main():
    sources = list((REPO_ROOT/'evaluation').rglob('*.py')) + list((REPO_ROOT/'analysis').glob('*.py'))
    for path in sources:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    assert sorted(p.relative_to(REPO_ROOT).as_posix() for p in REPO_ROOT.rglob('*.md') if '.git' not in p.parts) == ['README.md']
    for name in ('paper', 'data', 'results', 'docs', 'scripts', 'configs'):
        assert not (REPO_ROOT/name).exists(), f'Unexpected repository folder: {name}'
    for path in (REPO_ROOT/'website/index.html',):
        page = Links()
        page.feed(path.read_text(encoding='utf-8'))
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            if url.path:
                assert (path.parent/unquote(url.path)).exists(), link
            elif url.fragment:
                assert url.fragment in page.ids, link
    records = json.loads((ROOT/'results/provenance.json').read_text(encoding='utf-8'))
    for row in records:
        path = ROOT/row['release']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['release_sha256'], row['release']
    print(f'Validated {len(sources)} Python files, one README, website links, and {len(records)} unchanged saved-result files.')

if __name__ == '__main__':
    main()
