"""Generate the one-page brief an annotator reads before starting.

    PYTHONPATH=. .venv/bin/python -m tools.annotation_brief --out docs/annotation-brief.pdf

`docs/annotation-guideline.md` is the agreed rulebook and stays the authority:
it is the document DATA.md requires to exist before annotation begins, and the
place a new rule gets written down. This is not a second rulebook. It is the
subset a person needs in front of them while working, in the order they need it,
so that nobody annotates from memory of a document they skimmed once.

Where the two differ, the guideline wins, and that is stated on the sheet
itself rather than assumed.

reportlab is an optional dependency, declared in the `ml` group: generating a
briefing sheet is a development task and no serving path imports it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Final

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT: Final = Path("docs/annotation-brief.pdf")

INK: Final = colors.HexColor("#16191f")
SOFT: Final = colors.HexColor("#5b6472")
FAINT: Final = colors.HexColor("#8b95a3")
RULE: Final = colors.HexColor("#dde1e8")
ACCENT: Final = colors.HexColor("#1d4f91")

# The four types, with the tool's own colours so the sheet and the screen agree.
TYPES: Final = (
    ("1", "factual_assertion", colors.HexColor("#b9d6f2"), colors.HexColor("#0f3b6b")),
    ("2", "opinion", colors.HexColor("#f2cfc4"), colors.HexColor("#7d3417")),
    ("3", "prediction", colors.HexColor("#d9cdf0"), colors.HexColor("#45296e")),
    ("4", "attribution", colors.HexColor("#c6e2cb"), colors.HexColor("#1f5230")),
)


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=19,
            leading=22,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=2,
        ),
        "standfirst": ParagraphStyle(
            "standfirst",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=11,
            leading=15,
            textColor=SOFT,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=ACCENT,
            spaceBefore=13,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=10.5,
            leading=14.5,
            textColor=INK,
            spaceAfter=6,
        ),
        "quote": ParagraphStyle(
            "quote",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=10.5,
            leading=14.5,
            textColor=INK,
            leftIndent=12,
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.5,
            textColor=FAINT,
            spaceAfter=4,
        ),
        "cell": ParagraphStyle(
            "cell",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=9.5,
            leading=12.5,
            textColor=INK,
        ),
        "cellhead": ParagraphStyle(
            "cellhead",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=SOFT,
        ),
    }


def type_table(s: dict[str, ParagraphStyle]) -> Table:
    rows: list[list[Any]] = [
        [
            Paragraph("KEY", s["cellhead"]),
            Paragraph("TYPE", s["cellhead"]),
            Paragraph("WHAT IT IS", s["cellhead"]),
            Paragraph("EXAMPLE", s["cellhead"]),
        ]
    ]
    meaning = {
        "factual_assertion": "States something about the world that evidence could bear on. "
        "You are not judging whether it is true.",
        "opinion": "Evaluation or judgement. Cannot be checked, only agreed with.",
        "prediction": "About a future state, however confidently stated. Not checkable yet.",
        "attribution": "Reports that someone said or believes something. The checkable part "
        "is that they said it.",
    }
    example = {
        "factual_assertion": "Unemployment fell to 4.1% in March.",
        "opinion": "The policy was a disgrace.",
        "prediction": "Rates will rise again before June.",
        "attribution": "The minister said the deficit had halved.",
    }
    for key, name, _fill, ink in TYPES:
        label = ParagraphStyle(
            f"lbl{key}", parent=s["cell"], fontName="Helvetica-Bold", textColor=ink, fontSize=9
        )
        rows.append(
            [
                Paragraph(f"<b>{key}</b>", s["cell"]),
                Paragraph(name.replace("_", " "), label),
                Paragraph(meaning[name], s["cell"]),
                Paragraph(f"<i>{example[name]}</i>", s["cell"]),
            ]
        )

    table = Table(rows, colWidths=[11 * mm, 28 * mm, 67 * mm, 62 * mm], hAlign="LEFT")
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.4, RULE),
    ]
    for n, (_key, _name, fill, _ink) in enumerate(TYPES, start=1):
        style.append(("BACKGROUND", (1, n), (1, n), fill))
    table.setStyle(TableStyle(style))
    return table


def test_table(s: dict[str, ParagraphStyle]) -> Table:
    rows = [
        ["SENTENCE", "TYPE", "WHY"],
        ["The minister said the deficit had halved.", "attribution", "The minister would be wrong"],
        [
            "Unemployment fell to 4.1%, according to the ONS.",
            "factual assertion",
            "The paper would be wrong",
        ],
        ["Critics say the plan is unworkable.", "attribution", "The critics would be wrong"],
        [
            "Data published on Tuesday showed a fall.",
            "factual assertion",
            "The paper would be wrong",
        ],
    ]
    data: list[list[Any]] = [[Paragraph(c, s["cellhead"]) for c in rows[0]]]
    for row in rows[1:]:
        data.append(
            [
                Paragraph(f"<i>{row[0]}</i>", s["cell"]),
                Paragraph(f"<b>{row[1]}</b>", s["cell"]),
                Paragraph(row[2], s["cell"]),
            ]
        )
    table = Table(data, colWidths=[82 * mm, 36 * mm, 50 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
                ("LINEBELOW", (0, 1), (-1, -2), 0.4, RULE),
            ]
        )
    )
    return table


def rule(space: float = 7) -> HRFlowable:
    return HRFlowable(width="100%", thickness=0.6, color=RULE, spaceBefore=space, spaceAfter=space)


def build(out: Path) -> None:
    s = styles()
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Marking claims in news passages",
        author="Provenance",
    )
    story: list[Any] = []

    story.append(Paragraph("Marking claims in news passages", s["title"]))
    story.append(
        Paragraph(
            "You are reading 200 short passages from real news articles and pointing at the "
            "sentences that make a checkable claim. Roughly a minute or two each.",
            s["standfirst"],
        )
    )
    story.append(rule(4))

    story.append(Paragraph("What this is for", s["h2"]))
    story.append(
        Paragraph(
            "A program tries to find these same claims automatically. To know whether it works, "
            "somebody has to find them by hand first, and then the two get compared. That is what "
            "you are producing: the human answer the program gets marked against.",
            s["body"],
        )
    )
    story.append(
        Paragraph(
            "This is why it cannot be generated. A program graded against labels another program "
            "wrote is grading itself, and every number after that is worthless. Your disagreement "
            "with the machine is the whole point of the exercise.",
            s["body"],
        )
    )

    story.append(Paragraph("You are not deciding what is true", s["h2"]))
    story.append(
        Paragraph(
            "Nothing here asks whether a claim is correct. A false statement of fact is still a "
            "factual assertion. The question is only what <i>kind</i> of statement it is, never "
            "whether it holds up.",
            s["body"],
        )
    )

    story.append(Paragraph("The four types", s["h2"]))
    story.append(type_table(s))

    story.append(Paragraph("The rule that settles most cases", s["h2"]))
    story.append(
        Paragraph(
            "The hard boundary is between <b>attribution</b> and <b>factual assertion</b>, because "
            "a sentence can report speech and still contain a perfectly checkable fact. Apply this "
            "test:",
            s["body"],
        )
    )
    story.append(
        Paragraph(
            "<b>If the claim turned out to be false, would the article have been wrong, or would "
            "the speaker have been wrong?</b>",
            s["quote"],
        )
    )
    story.append(
        Paragraph(
            "Article wrong means it is an assertion. Speaker wrong means it is an attribution. "
            "Note that a bare source tag does not make a sentence an attribution: a paper that "
            "cites where a figure came from is still asserting the figure.",
            s["body"],
        )
    )
    story.append(Spacer(1, 3))
    story.append(test_table(s))

    story.append(Paragraph("A whole passage, worked through", s["h2"]))
    story.append(
        Paragraph(
            "This is passage 1 of the set. Eight spans, and the reasoning for each.", s["body"]
        )
    )
    worked = [
        (
            "Voting is underway in the Bangsamoro Autonomous Region...",
            "1",
            "Checkable. If false, the paper was wrong.",
        ),
        (
            "...more than 2.3 million people are eligible to vote",
            "1",
            "Mark only this clause. The 'seen as a crucial step' half is evaluation, so leave it.",
        ),
        ("There are 80 seats up for grabs in the elections...", "1", "Plain checkable fact."),
        ("These include 40 political party representatives...", "1", "Same."),
        (
            "The region's chief minister... will be elected by parliament...",
            "3",
            "Future state. Confidently stated, still not checkable yet.",
        ),
        (
            "Election officials said they were targeting a turnout of...",
            "4",
            "'said'. If the target is bogus, the officials were wrong, not the paper.",
        ),
        ("Barnaby Lo, reporting from Marawi, said 'voters are saying...'", "4", "Reported speech."),
        (
            "However, deadly violence marred the elections one day before...",
            "1",
            "The paper asserts it directly.",
        ),
    ]
    data: list[list[Any]] = [
        [
            Paragraph("TEXT ON SCREEN", s["cellhead"]),
            Paragraph("KEY", s["cellhead"]),
            Paragraph("WHY", s["cellhead"]),
        ]
    ]
    for text, key, why in worked:
        data.append(
            [
                Paragraph(f"<i>{text}</i>", s["cell"]),
                Paragraph(f"<b>{key}</b>", s["cell"]),
                Paragraph(why, s["cell"]),
            ]
        )
    table = Table(data, colWidths=[78 * mm, 11 * mm, 79 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
                ("LINEBELOW", (0, 1), (-1, -2), 0.4, RULE),
            ]
        )
    )
    story.append(table)

    story.append(Paragraph("Cases already decided, so you do not have to", s["h2"]))
    decided = [
        (
            "Passage with no claims at all",
            "Common and correct, especially in commentary. Press <b>0</b>. It counts as done, not "
            "as a skip. <b>Do not hunt for a claim to justify the passage</b> - a passage with "
            "nothing marked is a real data point and the program is scored on returning nothing "
            "there too.",
        ),
        (
            "Compound sentences",
            "Mark the shortest span that carries a whole claim. Two separable claims in one "
            "sentence means two spans.",
        ),
        (
            "Hedged attribution",
            "'Ministers are understood to be considering...' is <b>attribution</b>. The hedge "
            "names an unnamed source, which is still reported speech.",
        ),
        (
            "Numbers inside a judgement",
            "'A disgraceful 40% rise' - mark the figure's clause as a factual assertion and leave "
            "the evaluation out, rather than marking the whole sentence as opinion.",
        ),
        (
            "Negative existence",
            "'No evidence has emerged that...' is a <b>factual assertion</b> about the state of "
            "the evidence.",
        ),
        ("Questions", "Not claims, rhetorical or otherwise. Leave unmarked."),
        (
            "Headline-style fragments in the body",
            "Mark them if they assert. A missing verb does not disqualify them.",
        ),
        (
            "Where to start and stop a span",
            "Leave out the trailing full stop unless it falls inside the claim. Leave out leading "
            "narration ('Speaking on Tuesday, ...') unless it carries the attribution.",
        ),
    ]
    for heading, text in decided:
        story.append(
            KeepTogether(
                [
                    Paragraph(f"<b>{heading}.</b> {text}", s["body"]),
                ]
            )
        )

    story.append(Paragraph("Using the tool", s["h2"]))
    story.append(
        Paragraph(
            "Put your name in the box at the top first, so the work is attributable. Then "
            "select a sentence with the mouse and press <b>1</b>, <b>2</b>, <b>3</b> or "
            "<b>4</b>. <b>0</b> "
            "marks a passage as having no claims and moves on. <b>j</b> and <b>k</b> move forward "
            "and back, <b>u</b> removes the last span you added, and clicking <i>remove</i> in the "
            "list deletes a particular one.",
            s["body"],
        )
    )
    story.append(
        Paragraph(
            "Every span saves the moment you mark it. Closing the tab loses nothing, and reopening "
            "starts you at the first passage nobody has done.",
            s["body"],
        )
    )

    # The two warnings and the heading stay on one page: a rule split across a
    # page break is a rule half of the readers will not see.
    spoil: list[Any] = [
        Paragraph("Two things that would spoil the data", s["h2"]),
        Paragraph(
            "<b>Do not look at what the program produced</b> for a passage before marking it. "
            "If you have already seen its output for one, skip that passage and take another. "
            "Seeing the answer first is how a human marker drifts toward agreeing with a "
            "machine.",
            s["body"],
        ),
        Paragraph(
            "<b>Do not decide a new rule silently.</b> If a case is not covered here, type "
            "what you did in the notes box and carry on. The list of cases that needed a new "
            "rule is itself a finding, and a guideline quietly edited to match what was "
            "already marked measures nothing.",
            s["body"],
        ),
    ]
    story.append(KeepTogether(spoil))

    story.append(rule(9))
    story.append(
        Paragraph(
            "The full rulebook is docs/annotation-guideline.md and it is the authority. Where this "
            "sheet and that document differ, the document wins. If two of you disagree about a "
            "passage, say so rather than settling it quietly: disagreement between two careful "
            "readers is a result about the guideline, not noise.",
            s["small"],
        )
    )

    doc.build(story)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    build(args.out)
    sys.stdout.write(f"{args.out} ({args.out.stat().st_size // 1024} KiB)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
