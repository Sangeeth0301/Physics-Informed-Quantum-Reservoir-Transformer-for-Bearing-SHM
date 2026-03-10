
import os
import requests

# Paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
template_dir = os.path.join(base_dir, 'docs', 'templates')
os.makedirs(template_dir, exist_ok=True)

templates = {
    "Elsevier_Word_Template.docx": "https://legacyfileshare.elsevier.com/promis_misc/Research%20Article%20template%20and%20guidance.docx",
    "Elsevier_CAS_LaTeX.zip": "https://assets.ctfassets.net/o78em1y1w4i4/5uFmLZJTPDMAUjFnHRpjj8/6f19a979146eb93263763d87a894ab0d/els-cas-templates.zip",
    "elsarticle_classic.zip": "https://mirrors.ctan.org/macros/latex/contrib/elsarticle.zip"
}

def download_templates():
    print(f"Downloading official Elsevier templates to {template_dir}...")
    for name, url in templates.items():
        try:
            print(f"Fetching {name}...")
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            save_path = os.path.join(template_dir, name)
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"   -> Success: {name}")
        except Exception as e:
            print(f"   -> Failed to download {name}: {e}")

if __name__ == "__main__":
    download_templates()
