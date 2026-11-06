"""Check packaged code, local website references, and source-artifact hashes."""
import ast
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        self.links += [attrs[k] for k in ('href', 'src') if k in attrs]


def main():
    errors = []
    sources = list((ROOT/'code').rglob('*.py')) + list((ROOT/'scripts').glob('*.py'))
    for path in sources:
        try:
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except SyntaxError as error:
            errors.append(str(error))
    for path in [ROOT/'index.html', ROOT/'website/index.html']:
        page = Links()
        page.feed(path.read_text(encoding='utf-8'))
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            if url.path and not (path.parent/unquote(url.path)).exists():
                errors.append(f'Missing link from {path.name}: {link}')
            if not url.path and url.fragment and url.fragment not in page.ids:
                errors.append(f'Missing anchor: {link}')
    records = json.loads((ROOT/'configs/provenance.json').read_text(encoding='utf-8'))
    for row in records:
        path = ROOT/row['release']
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != row['release_sha256']:
            errors.append(f"Provenance mismatch: {row['release']}")
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Validated {len(sources)} Python files, local website links, and {len(records)} provenance entries.')
    print('Numerical agreement is checked separately by scripts/reproduce.py --strict.')


if __name__ == '__main__':
    main()
