# BRIEF

## The problem with the obvious version

The standard student project trains a transformer on a public fake news dataset, reports high nineties accuracy, and wraps it in Flask. That project fails for reasons that are visible to anyone who probes it for ninety seconds.

The headline accuracy is an artifact of dataset construction. In the most widely used corpus the true articles are almost entirely Reuters wire copy carrying a consistent dateline prefix, while the false articles come from a separate scrape with different tokenisation, different punctuation habits, and different average length. A classifier reaches ninety nine percent by learning publisher formatting. Point it at an article published this week and it degrades toward chance. The metric is real, the capability is not.

The deeper problem is conceptual. Veracity is not a property of a text string. Two articles can be word for word identical in style and differ entirely in truth value, which depends on facts about the world that no amount of surface text modelling can reach. A model that emits a truth verdict from text alone is making a category error, and shipping that verdict to users as a warning is worse than useless because it launders a stylistic guess as an epistemic judgement.

## What is being built instead

An evidence surfacing system. Given a claim or an article, it answers four questions that are actually answerable from data.

**What is being claimed?** Extract discrete checkable claims from free text, separating factual assertions from opinion, prediction, and rhetorical framing.

**Who else is reporting this?** Retrieve semantically related coverage from an indexed corpus and from live news retrieval, then classify each retrieved passage as supporting, refuting, or neutral toward the extracted claim using a natural language inference model.

**Where did this come from?** Resolve the publishing domain against a provenance registry recording ownership, registration age, corrections policy, and transparency signals, with the source of every field recorded so the user can audit the audit.

**How is it moving?** Measure circulation velocity and account diversity on open social platforms. A claim appearing across two hundred accounts created in the same fortnight is a different object than a claim spreading across long lived accounts with varied posting histories, and that difference is measurable without asserting anything about truth.

Rhetorical pattern detection sits alongside these as a secondary, clearly labelled signal. The fine tuned transformer flags spans exhibiting unsourced attribution, emotional loading, or manufactured urgency. It is presented as a writing observation, never as evidence of falsity, because prose style correlates with truth only weakly and the correlation is dataset specific.

## Why this framing is technically stronger

The reframe is not a retreat from the ML. It expands it. The original design needs one classifier. This needs claim extraction, dense retrieval, NLI stance classification, span level attribution, and a calibration layer that decides when to abstain. Every one of those is a component with an honest evaluation story, and none of them collapses when the input distribution shifts, because none of them is pretending to know something it cannot know.

It also produces a defensible answer to the hardest interview question about this project, which is some version of "how do you know it works". The answer is a documented distribution shift analysis and a retrieval quality measurement, not a single accuracy number.

## Users

**Primary: the reader mid scroll.** Encounters a claim, wants context in under three seconds without leaving the page. Needs a compact card, not a report. Will not read a paragraph of hedging.

**Secondary: the researcher or journalism student.** Wants the full evidence chain, the retrieval scores, the source registry entries, and the ability to export. Tolerates density and wants to see uncertainty made explicit.

**Tertiary: you, at the review.** Needs the system to be explainable end to end, with every number traceable to a measurement procedure written down before the number existed.

## Scope

In scope for the build.

- English language text. Multilingual is an explicit later phase, not a stretch goal quietly abandoned.
- Claim extraction, retrieval, NLI stance classification, source registry, circulation signal, rhetoric flags, calibration and abstention.
- Flask API, Postgres with pgvector, Redis, a Manifest V3 browser extension, a minimal web client for demonstration.
- A published out of distribution evaluation suite and a written analysis of where the system fails.

Out of scope, stated so the agent does not drift into building them.

- Image, video, or audio verification. Different problem, different literature, different compute budget.
- Real time firehose ingestion at scale. The circulation signal samples; it does not attempt coverage.
- Any automated moderation, takedown, reporting, or content ranking action.
- Any user account system beyond an API key for rate limiting. No profiles, no history, no personalisation.
- Any claim of completeness. Absence of retrieved evidence is reported as absence of retrieved evidence, never as evidence of absence.

## The line that defines the product

The system surfaces what can be established about a claim's provenance and its relationship to other reporting. It stops there. The judgement stays with the reader, and the interface is designed to make that handover explicit rather than to obscure it behind a confidence percentage.
