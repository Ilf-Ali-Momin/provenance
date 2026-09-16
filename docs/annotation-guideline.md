# Annotation guideline: checkable claims in news text

This is the written guideline DATA.md requires, to be agreed **before**
annotation begins. It exists for a specific reason stated there: building the
set "forces the annotation guideline to be explicit about the assertion versus
opinion boundary, which is the hard part of the task".

Everything below is a rule you can disagree with. Disagree before you annotate,
not after, and change the rule rather than the annotations — a guideline edited
to match what was already marked measures nothing.

## What you are producing

200 passages from the out-of-distribution suite, each marked with zero or more
spans. For each span: a start offset, an end offset, and one of four types.

The extractor is evaluated against this set on **assertion identification**
(precision and recall, a match counted when predicted and gold spans overlap by
more than half) and on the **four-way type confusion matrix**. Acceptance is
recall above 0.80 with precision above 0.70 (EVALUATION.md §3).

## The four types

### `factual_assertion`

States something about the world that could in principle be checked against
evidence. "In principle" is doing real work: you are not judging whether it *is*
true, or whether checking is practical, only whether evidence could bear on it.

> Unemployment fell to 4.1% in March.
> The bridge carries 40,000 vehicles a day.
> No member of the committee declared an interest.

### `opinion`

Evaluation, preference, or judgement. Cannot be checked, only agreed with.

> The policy was a disgrace.
> This is the most important election in a generation.
> The design is ugly.

### `prediction`

Concerns a future state, however confidently stated, and is therefore not
checkable at the time of writing.

> Rates will rise again before June.
> The scheme is expected to cost £4bn.

### `attribution`

Reports that someone said, believes, or claims something. The checkable content
is **that they said it**, not that it is true.

> The minister said the deficit had halved.
> According to the report, emissions rose.
> Critics argue the plan is unworkable.

## The boundary that matters most

**Attribution beats factual_assertion whenever the sentence reports speech**,
even when the reported content is itself perfectly checkable. This is the single
rule most likely to be applied inconsistently, and inconsistency here is the
largest error source in stance classification downstream (EVALUATION.md names
attributed claims as its own stress subset).

> The minister said the deficit had halved.

is `attribution`, not `factual_assertion`, even though "the deficit halved" is
checkable. What the article asserts is the saying.

**But: a bare source tag does not make a sentence an attribution.**

> Unemployment fell to 4.1% in March, according to the statistics office.

The paper is asserting the figure and citing where it came from. Mark
`factual_assertion`. The test to apply: **if the claim turned out to be false,
would the article have been wrong, or would the speaker have been wrong?** If
the article would have been wrong, it is an assertion.

Applying that test:

| Sentence | Type | Why |
| --- | --- | --- |
| The minister said the deficit had halved. | `attribution` | The minister would be wrong |
| Unemployment fell to 4.1%, according to the ONS. | `factual_assertion` | The paper would be wrong |
| Critics say the plan is unworkable. | `attribution` | The critics would be wrong |
| Data published on Tuesday showed a fall. | `factual_assertion` | The paper would be wrong |

## Other hard cases, decided in advance

**Compound sentences.** Mark the shortest span carrying a whole claim. If one
sentence contains two separable checkable claims, mark two spans.

**Hedged assertions.** "Ministers are understood to be considering…" — mark
`attribution`. The hedge names an unnamed source, which is still reported
speech.

**Numbers in opinion.** "A disgraceful 40% rise" contains a checkable figure
inside a judgement. Mark the figure's clause `factual_assertion` and leave the
evaluation unmarked, rather than marking the whole sentence `opinion`.

**Negative existence.** "No evidence has emerged that…" is a
`factual_assertion` about the state of evidence.

**Questions and rhetorical questions.** Not claims. Leave unmarked.

**Headline-style fragments** inside body text. Mark if they assert; the absence
of a finite verb does not disqualify them.

**Passage with no claims at all.** Common and correct, especially in
commentary. Mark nothing. **Do not hunt for a claim to justify the passage's
inclusion** — a passage with zero spans is a data point, and the extractor is
scored on returning zero there too.

## Span conventions

- Offsets are character offsets into the passage text exactly as supplied,
  counting from zero.
- Do not include the trailing full stop unless it falls inside the claim.
- Do not include leading narration ("Speaking on Tuesday, …") unless it carries
  the attribution.
- Do not correct, normalise, or reformat the text. You are pointing at it.

## Procedure

1. Read the whole passage before marking anything.
2. Mark spans in reading order.
3. Where a case is not covered above, **do not decide silently.** Record it in
   the `notes` field, finish the passage, and add the rule here afterwards with
   the date. The list of cases that needed a new rule is itself a result.
4. Do not look at extractor output while annotating. If you have already seen
   what the model produced for a passage, skip that passage and take another.

## What is deliberately not annotated

Truth. No span is marked true or false, and no field exists to record it. This
set trains and evaluates nothing about veracity (D001); it measures whether the
extractor finds the same *checkable* statements a human finds, and labels them
with the same type.
