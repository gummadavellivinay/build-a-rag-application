"""Generate only synthetic corpus, graph, inventory and OCR fixtures; never fetch real listings."""

import io
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pymupdf

root = Path(__file__).resolve().parents[1] / "data"
corpus = root / "corpus"
corpus.mkdir(parents=True, exist_ok=True)
manifest, nodes, edges, inventory = [], [], [], []


def node(id, name, kind, aliases=None):
    if id not in {n["id"] for n in nodes}:
        nodes.append(dict(id=id, name=name, type=kind, aliases=aliases or [], tenant="demo"))


def edge(source, target, relation, evidence, document):
    edges.append(
        dict(
            id=f"E{len(edges) + 1:03}",
            source=source,
            target=target,
            relation=relation,
            evidence=evidence,
            document=document,
            tenant="demo",
        )
    )


for i in range(1, 7):
    listing = f"L-{i:03}"
    broker = f"B-{(i - 1) // 2 + 1:02}"
    office = f"O-{(i - 1) // 4 + 1:02}"
    area = "N-01" if i <= 3 else "N-02"
    price, beds = 400000 + i * 25000, 2 + i % 3
    name = f"{100 + i} Cedar Lane"
    text = (
        f"SYNTHETIC DEMO LISTING. {listing}: {name}. Asking price USD {price}; {beds} bedrooms. "
        f"Features: {'solar panels and a private garden' if i % 2 else 'step-free entry and an elevator'}. "
        f"Broker {broker} markets the listing. Neighborhood {area}. "
        "Listing brochure date 2026-10-01. This brochure does not establish current availability.\n"
    )
    filename = f"corpus/{listing}.txt"
    (root / filename).write_text(text)
    manifest.append(dict(path=filename, listing_id=listing, language="en"))
    node(listing, name, "Listing", [listing])
    node(broker, f"Broker {(i - 1) // 2 + 1}", "Broker")
    node(office, f"Agency office {(i - 1) // 3 + 1}", "Office")
    node(area, "Cedar District" if i <= 3 else "Lake District", "Neighborhood")
    amenity = f"A-{i:02}"
    docid = f"D-{i:02}"
    node(amenity, f"Park {i}", "Amenity")
    node(docid, f"Brochure {listing}", "Document")
    edge(listing, broker, "MARKETED_BY", f"Broker of record for {listing} is {broker}.", filename)
    edge(
        broker,
        office,
        "WORKS_AT",
        f"Broker {broker} is affiliated with {office}.",
        "corpus/relationships.txt",
    )
    edge(listing, area, "LOCATED_IN", f"{listing} lies in {area}.", filename)
    edge(
        listing,
        amenity,
        "NEAR",
        f"{listing} is near Park {i}; exact distance not documented.",
        "corpus/relationships.txt",
    )
    edge(listing, docid, "DOCUMENTED_BY", f"Brochure {filename} documents {listing}.", filename)
    inventory.append(
        dict(
            listing_id=listing,
            tenant="demo",
            status="active" if i % 2 else "pending",
            asking_price=price,
            currency="USD",
            updated_at="2026-10-15T12:00:00Z",
        )
    )
relationships = "SYNTHETIC GRAPH SOURCE\n" + "\n".join(e["evidence"] for e in edges)
(corpus / "relationships.txt").write_text(relationships)
manifest.append(dict(path="corpus/relationships.txt", language="en"))

# A Spanish searchable PDF and French/English raster fixtures exercise independent paths.
pdf = pymupdf.open()
p = pdf.new_page()
p.insert_text(
    (50, 70), "FICHA SINTETICA L-001: La vivienda tiene paneles solares y un jardin privado.", fontsize=12
)
pdf.save(corpus / "spanish.pdf")
manifest.append(dict(path="corpus/spanish.pdf", language="es", listing_id="L-001"))
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 30)
except OSError:
    font = ImageFont.load_default(size=30)
image = Image.new("RGB", (1600, 320), "white")
ImageDraw.Draw(image).multiline_text(
    (40, 50),
    "SYNTHETIC SCREENSHOT L-002\nAccessible entrance. Elevator. Two parking spaces.",
    fill="black",
    font=font,
    spacing=30,
)
image.save(corpus / "screenshot.png")
manifest.append(dict(path="corpus/screenshot.png", language="en", listing_id="L-002"))
image_fr = Image.new("RGB", (1600, 320), "white")
ImageDraw.Draw(image_fr).multiline_text(
    (40, 50),
    "DOCUMENT SYNTHETIQUE L-003\nLa maison dispose de deux chambres et d'un jardin.",
    fill="black",
    font=font,
    spacing=30,
)
pdf = pymupdf.open()
p = pdf.new_page(width=800, height=160)
buf = io.BytesIO()
image_fr.save(buf, format="PNG")
p.insert_image(p.rect, stream=buf.getvalue())
pdf.save(corpus / "french-scan.pdf")
manifest.append(dict(path="corpus/french-scan.pdf", language="fr", listing_id="L-003"))
(root / "manifest.json").write_text(json.dumps(manifest, indent=2))
# Text-only manifest supports a first cloud smoke test before Tesseract is installed.
(root / "manifest-text.json").write_text(
    json.dumps([r for r in manifest if r["path"].endswith((".txt", "spanish.pdf"))], indent=2)
)
(root / "graph.json").write_text(json.dumps(dict(nodes=nodes, edges=edges), indent=2))
(root / "inventory.json").write_text(json.dumps(inventory, indent=2))
print(f"Created {len(manifest)} corpus documents, {len(nodes)} nodes and {len(edges)} edges. All synthetic.")
