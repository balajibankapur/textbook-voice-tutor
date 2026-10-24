# Images for the article

Both `.svg` (source, editable) and `.png` (Medium-ready, 1400px wide) are kept — Medium accepts both but PNG uploads most reliably.

### Full-width (section-ending) diagrams

| File | Where it appears in `article.md` |
|---|---|
| `architecture.png` | End of "Architecture Summary" — full end-to-end pipeline |
| `ab-latency-cost.png` | End of "A/B: Chat API vs. the Real Decisions API" — bar chart |
| `cache-ab.png` | End of "A/B With and Without Cache" — latency comparison |

### Smaller inline diagrams (between sections)

| File | Where it appears |
|---|---|
| `ingestion-flow.png` | Inside "Ingestion (one-time per chapter)" — horizontal 6-step flow |
| `policy-state-machine.png` | After the three "design points" callouts — gate branches |
| `jev-types.png` | After the four-primitives table — noul / choice / score cards |
| `cache-flow.png` | Inside the Q&A cache "Pattern" subsection — match/miss flow |

### Hero (needs you to add)

Take a screenshot of the Streamlit app in voice mode with the status pill showing 🟢 **"Ready — press the mic to speak"** and the Plants chapter selected. Save as **`hero.png`** here. On Medium that becomes the article's cover.

### Regenerating PNGs from SVG source

```bash
# Needs librsvg (installed by brew install tesseract etc. — may already be present)
cd docs/images
for f in *.svg; do
  rsvg-convert "$f" -o "${f%.svg}.png" -w 1400
done
```

### Pasting into Medium

- Upload PNGs directly (drag-and-drop into the editor)
- Medium will downsize for display but keep the 1400px source for retina/high-DPI
- SVG can also be dropped in; browser support in Medium's editor is good but PNG is more consistent

Each image is standalone with its own title, so they read well whether readers pause on them or scroll past.
