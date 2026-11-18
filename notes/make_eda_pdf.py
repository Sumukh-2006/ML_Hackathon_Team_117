"""Render the PathSnap EDA report to PDF.

Typography matches the course documents: US Letter, Times New Roman, 11 pt body,
23 pt bold title. Every number in the report is reproduced by notes/verify_eda.py,
notes/facts.py or notes/candidates.py.

Run: python notes/make_eda_pdf.py
Out: submissions/EDA_Report.pdf
"""
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "submissions", "EDA_Report.pdf")

BODY = ParagraphStyle("body", fontName="Times-Roman", fontSize=11, leading=14.2,
                      alignment=TA_LEFT, spaceAfter=5)
BULL = ParagraphStyle("bull", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3,
                      bulletFontName="Times-Roman")
H1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=11, leading=14,
                    spaceBefore=12, spaceAfter=5, keepWithNext=1)
TITLE = ParagraphStyle("title", fontName="Times-Bold", fontSize=22, leading=26,
                       spaceAfter=4)
SUB = ParagraphStyle("sub", fontName="Times-Bold", fontSize=11, leading=14,
                     spaceAfter=10)
CAP = ParagraphStyle("cap", fontName="Times-Italic", fontSize=9, leading=11.5,
                     spaceBefore=3, spaceAfter=8, textColor=colors.HexColor("#333333"))


def P(t):
    return Paragraph(t, BODY)


def B(t):
    return Paragraph(t, BULL, bulletText="\u2022")


def H(t):
    return Paragraph(t, H1)


def table(rows, widths, size=9.5):
    data = [[Paragraph(c, ParagraphStyle("c", fontName="Times-Bold" if i == 0 else "Times-Roman",
                                         fontSize=size, leading=size + 2.4))
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.9, colors.black),
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


S = []

S += [Paragraph("PATHSNAP: EXPLORATORY DATA ANALYSIS", TITLE),
      Paragraph("Team 117", SUB)]

S += [H("1. Purpose")]
S += [P("This document records the exploratory analysis of the PathSnap dataset: what the raw "
        "files contain, how they relate to one another, what the data quality problems are, and "
        "which assumptions follow from them. Every figure quoted here was measured on the files "
        "themselves, not taken on trust, and is reproducible from the scripts held with the "
        "project.")]

S += [H("2. Dataset Overview")]
S += [P("The dataset contains GPS trajectories recorded by vehicles, together with the road "
        "network that each trajectory must be matched against. The package is organised by "
        "trip. Each trip is a directory holding four files: the GPS fixes, the nodes of the "
        "road network, the directed road segments joining those nodes, and the reference route "
        "the vehicle actually took.")]
S += [P("Ninety trips are provided: eighty for training and ten for development-stage "
        "evaluation. A further ten trips are held back for final evaluation and are not part "
        "of the package. Each trip is self-contained and carries its own road network; no "
        "network is shared between trips.")]

S += [H("3. Raw File Types")]
S += [table([
    ["File", "Rows", "Columns", "What it holds"],
    [".track", "GPS fixes", "3", "longitude, latitude, time in seconds — one row per fix, in time order"],
    [".nodes", "road nodes", "2", "longitude, latitude — the 0-based row index is the node id"],
    [".arcs", "directed segments", "2", "from_node_id, to_node_id — the 0-based row index is the arc id"],
    [".route", "reference path", "1", "arc_id values, in the order the vehicle traversed them"],
], [0.65 * inch, 1.05 * inch, 0.62 * inch, 4.3 * inch])]
S += [Paragraph("No file carries a header. All values are numeric and whitespace-separated. "
                "Every trip provides all four files; none are missing and none are empty.", CAP)]

S += [H("4. How the Files Relate")]
S += [P("The four files occupy four different index spaces and do not correspond by row. A "
        "trip typically has far more fixes than route arcs and roughly twice as many arcs as "
        "nodes; the counts are independent. There are exactly three links between the files, "
        "and they run in one direction:")]
S += [B("<b>.route</b> values are <b>arc ids</b>, so each value selects a row of <b>.arcs</b>."),
      B("<b>.arcs</b> holds node ids, so each row selects two rows of <b>.nodes</b>."),
      B("<b>.track</b> carries no ids at all. A fix is connected to the network only by "
        "measuring distance to segments; there is no join key.")]
S += [P("Consequently the row index is the identifier throughout: node ids are node row "
        "numbers, arc ids are arc row numbers, and neither appears as a column.")]

S += [H("5. Dataset Inventory")]
S += [table([
    ["Quantity", "Total", "Per trip (min / median / max)"],
    ["Trips shipped", "90", "—"],
    ["Files", "360", "4"],
    ["GPS fixes", "227,577", "262 / 1,496 / 19,313"],
    ["Road nodes", "2,024,174", "299 / 9,305 / 155,003"],
    ["Road segments", "4,061,945", "649 / 18,762 / 307,392"],
    ["Reference route segments", "34,813", "59 / 272 / 2,266"],
], [2.25 * inch, 1.15 * inch, 3.2 * inch])]
S += [P("The reference route is a small fraction of its own network: a median of 1.39% of the "
        "segments, ranging from 0.20% to 18.3% across trips. Total package size is 94.1 MB.")]

S += [H("6. Sampling and Timing")]
S += [P("The recording rate is nominally one fix per second, and this holds for most of the "
        "data but not evenly. Pooled across the whole dataset, 96.68% of the 227,487 steps "
        "between consecutive fixes are exactly one second (219,934 steps). The distribution "
        "across trips is strongly two-sided rather than uniformly degraded:")]
S += [B("24 of the 90 trips have no irregular step at all."),
      B("The median trip is 99.1% regular, and the 75th percentile is 100%."),
      B("The worst trip is 53.1% regular."),
      B("15 trips contain 80% of all irregular steps.")]
S += [P("Thirteen steps in the entire dataset are longer than thirty seconds, and eight are "
        "longer than ten minutes. In seven of those eight, the two fixes bracketing the gap "
        "are less than 50 m apart, so these are pauses in recording rather than missed "
        "journeys. The extremes make an important point about derived quantities: trip "
        "00000090 consists of 536 fixes spanning 970 s and 8.58 km, a mean of 31.8 km/h, but "
        "its final row is a single fix recorded 42.5 hours later, 3.3 m from its predecessor. "
        "Dividing distance by wall-clock duration therefore reports the trip as 42.8 hours "
        "long at 0.20 km/h. Trip duration and travel speed must be computed from consecutive "
        "fixes with the gap handling made explicit, never from first and last timestamps.")]
S += [P("Timestamps are offsets in seconds and carry no calendar information: no date, no "
        "time zone and no time of day. Their origin is arbitrary per trip, beginning anywhere "
        "between 1 s and 693 s. Ordering is reliable — no trip contains a non-increasing "
        "timestamp — but absolute time is unavailable.")]

S += [H("7. Geographical Extent and Projection")]
S += [P("The trips are spread across the world: 27 countries, longitudes from -100.3° to "
        "137.2°, and latitudes from -34.6° to 61.7°. The combination of UTM zone and "
        "hemisphere gives 25 distinct cases.")]
S += [P("A single projected coordinate system is therefore not valid for this dataset. Any "
        "one projection imposes scale error that grows with distance from its central "
        "meridian and with latitude, and the spread here is far too large to absorb it. The "
        "approach taken is one projection per trip, chosen from the trip's own centroid, which "
        "keeps scale error within 0.1% inside each trip's extent. Point-to-point distances are "
        "additionally computed geodesically where a projection is not needed. Coordinate "
        "order was verified on every file: longitude in column 0, latitude in column 1.")]

S += [H("8. Road Network Structure")]
S += [P("Each network is a directed graph. A road that may be driven in both directions is "
        "stored twice, as an arc (u, v) and a separate arc (v, u) with identical geometry. "
        "These are distinct, legal arcs, not duplicates, and the second is referred to as the "
        "reverse twin of the first. Their prevalence depends on what is counted:")]
S += [table([
    ["Population", "Arcs with a reverse twin"],
    ["All arcs in all ninety networks (4,061,945)", "92.3%"],
    ["Reference route arcs, pooled (34,813)", "62.1%"],
    ["Median trip's reference route", "70%"],
], [3.9 * inch, 2.0 * inch])]
S += [P("The consequence is that geometry alone cannot identify a route arc. A twin lies at "
        "exactly the same distance from any fix, so a distance-based ranking produces an exact "
        "tie and any tie-breaking rule decides the outcome. Direction of travel, estimated "
        "from the fixes either side of the current one, separates them. Measured on the "
        "142,306 fixes whose nearest reference arc has a twin, heading selects the true "
        "direction 85.2% of the time using a one-fix window, rising to 90.6% with a five-fix "
        "window. Direction is therefore necessary but not sufficient, and the residual 9-15% "
        "is a floor on achievable accuracy that no amount of distance modelling removes.")]
S += [P("Segment geometry is unevenly scaled: the median arc is 25.7 m long, 5.1% are shorter "
        "than 5 m, and the longest is 9,327 m. A further 106 arcs are of zero length, spread "
        "across 32 trips; none appears in a reference route, so they do not affect labels, but "
        "they will break any calculation of bearing or speed that divides by segment length.")]

S += [H("9. Candidate Generation")]
S += [P("A fix is associated with segments by radius search. Coverage was measured against "
        "the reference route, taking a hit to mean the reference arc appears in the candidate "
        "set:")]
S += [table([
    ["Radius", "K = 10", "Reference arc missing from candidates"],
    ["25 m", "10", "5.15% of fixes"],
    ["50 m", "10", "3.81% of fixes"],
    ["75 m", "10", "3.59% of fixes"],
], [0.9 * inch, 0.8 * inch, 3.2 * inch])]
S += [P("Most of the residual is not a search failure. 2.99% of fixes lie more than 50 m from "
        "their own reference arc and so cannot be captured by any radius of that size; "
        "restricting to fixes that are within 50 m of their own arc, the miss rate at 50 m and "
        "K = 10 falls to 1.12%, a recall of 98.9%. Raising the radius beyond 50 m recovers "
        "little, because the limiting factor becomes the number of candidates retained rather "
        "than the search distance. Fixes sit a median 4.7 m from their own arc, with a 95th "
        "percentile of 14.5 m; across trips the median ranges from 3.3 m to 138.3 m, so the "
        "residual is a property of individual trips and should not be pooled.")]

S += [H("10. Reference Routes and Their Limitations")]
S += [P("The reference route is the label against which matching is scored, and it is not "
        "uniformly reliable. The clearest case is trip 00000002, where 14.0% of fixes lie more "
        "than 30 m from any reference arc, the final 7.7% of the drive is covered by no "
        "reference arc at all, and the worst fix is 566 m from the nearest one. No other trip "
        "shows an uncovered tail longer than 10%, but the case is enough to establish that a "
        "reference route may be a partial cover rather than a complete one.")]
S += [P("Two further properties matter when scoring. First, 10 trips traverse the same arc "
        "more than once, up to three times, so a reference route is a walk rather than a "
        "simple path and any evaluation code that assumes each arc appears once will "
        "mis-handle them. Second, the reference route is itself noisy relative to the fixes: "
        "the median fix-to-arc residual is 4.7 m, which sets a floor on what any matcher can "
        "achieve against these labels. Accuracy figures should be reported with the affected "
        "trips named, not silently dropped.")]

S += [H("11. Data Quality Issues")]
S += [table([
    ["Issue", "Extent", "Consequence"],
    ["Reverse twins", "92.3% of arcs", "distance-only matching ties; direction required"],
    ["Irregular sampling", "15 trips carry 80% of it", "time is not an index on those trips"],
    ["Recorder pauses over 10 min", "8 steps in 6 trips", "transition scoring must be suspended"],
    ["Zero-length arcs", "106 arcs in 32 trips", "bearing and speed undefined there"],
    ["Route repeat visits", "10 trips, max 3", "evaluation must tolerate repeats"],
    ["Partial label cover", "trip 00000002 (14.0% of fixes)", "caps the attainable score"],
    ["Missing or duplicate fixes", "none", "0 NaN, 0 inf, 0 duplicate fixes"],
    ["Non-increasing timestamps", "none", "0 across all 227,577 fixes"],
], [1.85 * inch, 1.75 * inch, 2.7 * inch])]

S += [H("12. Assumptions and Decisions")]
S += [B("<b>Projection.</b> One projected system per trip, derived from the trip's own "
        "centroid; a single global projection was rejected because the dataset spans 25 UTM "
        "zone and hemisphere cases."),
      B("<b>Direction.</b> Heading estimated from a five-fix window is used to separate "
        "reverse twins; a one-fix window is measurably worse (85.2% against 90.6%)."),
      B("<b>Gaps.</b> Any step longer than the transition threshold is treated as a session "
        "boundary, not a movement, and no speed constraint is applied across it."),
      B("<b>Labels.</b> Reference routes are treated as reference information rather than "
        "verified physical ground truth, and trips with degraded coverage are reported "
        "separately rather than removed."),
      B("<b>Nodes.</b> Node coordinates are taken as given; no snapping or smoothing of the "
        "network is applied, since the network is the reference surface, not an observation.")]

S += [H("13. Conclusions")]
S += [P("The dataset is structurally clean and semantically difficult. Format, indexing and "
        "file availability present no obstacle: all 360 files parse, no fix is missing or "
        "duplicated, and every index is valid. The difficulty is concentrated in three "
        "places — GPS noise of the same order as segment length, a road graph in which most "
        "arcs are geometrically ambiguous, and a small number of trips whose timing or labels "
        "are unreliable in ways that mimic real driving.")]
S += [P("The practical consequences are that geometry alone cannot decide a match, that "
        "timing must be handled per step rather than per trip, and that accuracy must be "
        "reported with its denominators stated. A candidate set of 50 m with ten candidates "
        "per fix recovers 98.9% of reachable reference arcs and is the point beyond which "
        "wider search stops paying.")]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc = SimpleDocTemplate(OUT, pagesize=letter,
                            leftMargin=1.0 * inch, rightMargin=1.0 * inch,
                            topMargin=0.95 * inch, bottomMargin=0.95 * inch,
                            title="PathSnap: Exploratory Data Analysis",
                            author="Team 117")
    doc.build(S)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
