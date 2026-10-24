# Images for the article

| File | Where to use it |
|---|---|
| `architecture.svg` | End of the "Architecture Summary" section — replaces the ASCII diagram |
| `ab-latency-cost.svg` | End of "A/B: Chat API vs. the Real Decisions API" section |
| `cache-ab.svg` | End of "A/B With and Without Cache" section |

**Hero image (not generated — add yours):**
Take a screenshot of the Streamlit app in voice mode with the status pill showing 🟢 **"Ready — press the mic to speak"** and the Plants chapter selected. Save as `hero.png` here. On Medium, that becomes the article's cover image.

**Converting SVG → PNG for Medium:** Medium accepts SVG uploads directly in modern browsers. If you need a PNG for any reason:

```bash
# Needs librsvg: brew install librsvg
rsvg-convert docs/images/architecture.svg -o docs/images/architecture.png -w 1600
rsvg-convert docs/images/ab-latency-cost.svg -o docs/images/ab-latency-cost.png -w 1600
rsvg-convert docs/images/cache-ab.svg -o docs/images/cache-ab.png -w 1600
```
