# Project website

Static HTML, CSS, and JavaScript. No build step or external frontend dependencies.

From the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Open http://localhost:8000/website/.

Build the local download files with `python scripts/package_downloads.py`. The source archive excludes generated tables, installed dataset images, and download archives; rebuild the downloads if serving the website from an unpacked source archive.

`data.js` contains the printed paper tables (see `docs/RESULTS.md` for numerical discrepancies) and a small set of released example probes. Regenerate it after updating saved results with `python scripts/build_website_data.py`. The main figure is extracted from page 1 of `paper/paper.pdf`.

The page links to the sibling `paper/` and `docs/` directories. Serve or publish the repository root, not `website/` alone. The root `index.html` redirects to this page. Code and dataset links download local archives. No organization or public destination is configured. Once published, replace the code download URLs and the release-note spans with the actual repository and star links. No remote repository or website is created by these files.
