# Legitimate interests assessment: the evaluation corpora

DATA.md and SECURITY.md both state that the lawful basis for the evaluation
corpus is legitimate interest for research, "recorded with a documented
balancing test". `data/MANIFEST.yaml` required a path to that test and carried
`OPEN OBLIGATION: not yet written` in its place. This is that document.

It is an internal assessment by the project, not legal advice, and it is
written to be falsifiable: where a safeguard the assessment relies on is not
actually in place, it says so and says what that costs the conclusion.

## 1. What is processed, and whose data

Two corpora, with different profiles.

**The ISOT style mirror** (`data/MANIFEST.yaml`, `isot_mirror`). Third party
news text, 44,267 documents, held only in the local Hugging Face cache for the
duration of the ablations. It contains the names, quotations and reported
conduct of people who appear in news coverage — politicians, officials, people
caught up in events — plus journalists as authors. This project does not train
on it (`training_use: prohibited`, D010); it is measured and discarded.

**The out-of-distribution suite** (`ood_suite_*`). Articles collected by this
project from named publishers, stripped of publisher markers and bylines before
storage (D018), then frozen. It contains the same category of data subject:
people written about in the news. Author names are removed as a side effect of
the leakage controls rather than as a privacy measure, which is worth stating
plainly — the two happen to coincide here.

No special category data is sought. None is excluded either: news coverage
routinely reports health, political opinions, criminal allegations and
religious affiliation, so the corpora certainly contain Article 9 data about
some individuals, incidentally and without any way to filter it in advance.
This is the single most uncomfortable fact in this assessment and it is not
resolved by anything below; it is the main reason the retention limits in §5
are short and the deletion procedure in §6 exists.

## 2. Purpose test: is the interest legitimate?

The purpose is to establish whether a widely reproduced accuracy figure
measures what it claims to measure. The answer, measured rather than asserted
(`docs/leakage-analysis.md`), is that it does not: 81.8% of a 50%-chance task
survives the destruction of every content word, and the resulting model sorts
2026 news by publisher type and political lean.

That is a legitimate interest, and a third party one as well as a private one.
Systems of this kind are proposed for deciding what people are allowed to read.
Evidence that they are separating house style guides rather than truth from
falsehood is of direct benefit to the people such systems would be applied to,
which includes the data subjects here. The interest is not commercial; nothing
is sold, deployed or redistributed.

The counter-consideration is honest: "research" is the most abused lawful basis
in this field, and calling a project research does not make its processing
proportionate. The rest of this document is what distinguishes the two.

## 3. Necessity test: is the processing necessary?

Necessary for the leakage analysis, yes. The finding is a claim about a
specific real corpus. It cannot be established on synthetic text, because the
question is precisely whether *that corpus's* assembly produced the signal. A
smaller sample would weaken the result but not change its character; the sample
sizes here are set by what the ablations need, not by what was available.

Necessary for the out-of-distribution suite, yes, with one qualification. The
suite exists because a claim about generalisation cannot be tested on the
training distribution. It must be real, recent, and outside the corpus. But the
suite does not need to be large, and it is not: hundreds of documents, not
tens of thousands, and capped per publisher and per month (D022, D024).

Not necessary: retaining article text after the analysis that needs it. That is
what §6 addresses, and until it is implemented this test's necessity limb is
only satisfied prospectively.

## 4. Balancing: do the individuals' interests override?

**Toward proportionate.** Every document was published deliberately, by a
professional publisher, for public consumption. No private, hidden, or
inferred data is collected. No data subject is profiled: the unit of analysis
is the publisher and the document, never the person named inside it, and no
index by personal name exists or is built. Nothing is redistributed (DATA.md).
No decision of any kind is made about any individual — the system is
constitutionally incapable of producing a verdict about a person or a claim
(README.md, D001), which removes the entire category of harm that automated
decision-making rules exist to address. Robots directives and rate limits are
honoured, and a refusal ends collection from that host rather than being routed
around (D019).

**Toward disproportionate.** Reasonable expectations are the weak point. A
person quoted in a regional newspaper in March expects to be read; they do not
necessarily expect to be a row in a machine learning corpus on someone's
laptop. That gap is real and is not closed by the text having been public.
Incidental Article 9 data (§1) sharpens it. Aggregation is a genuine harm even
when each input is public, and this project aggregates.

**Where that lands.** The interest carries, but narrowly, and only because the
processing is small, non-decisional, non-redistributed, and time-limited. It
would not carry at a larger scale, and it would not carry at all if the output
made claims about individuals. Two of the safeguards the balance leans on are
not in place today (§5), so the honest statement is that the balance is met
*subject to* those, not independently of them.

## 5. Safeguards, and which are real

| Safeguard | Status |
| --- | --- |
| No article text in logs or error output | in place, tested |
| No article text in git (`data/` gitignored) | in place |
| Publisher markers and bylines stripped before storage | in place (D018) |
| Aggregate-only social data, no identifier retained | in place |
| No redistribution | in place |
| No verdict about any person or claim | in place, structural |
| Collection honours robots and refusals | in place (D019) |
| Encrypted storage in an EU region | **not in place** |
| Documented deletion procedure | **see §6** |

The storage gap is stated rather than worked around. The corpora sit on a
single developer machine, not in an EU-region encrypted store, so DATA.md's
description of where this data lives is currently aspirational. Disk encryption
on the host is not a substitute for the control as written. Anyone reading the
manifest should know the difference between what the policy says and what the
filesystem does.

## 6. Retention and deletion

The mirror is cache-only and is deleted with the cache. The frozen suite is
retained for the evaluation it supports, which is the project's lifetime and no
longer.

Deletion is implemented rather than promised: `eval/ood_collection/erase.py`
removes a corpus directory and every derived text artefact, and writes a
receipt naming what was removed and when. See `--help` for the exact command.
The receipt exists so a deletion claim can be checked rather than believed.

## 7. Outcome, and what would reverse it

**Outcome: proceed, at this scale, with the gaps above stated in the manifest
rather than closed by assertion.**

Reversing conditions, any one of which flips this assessment:

- The output begins to characterise individuals, or any claim, as true or
  false. This is prohibited structurally, so it would be a change of project.
- The corpus is redistributed, published, or shared outside the project.
- The corpus grows past the scale the evaluation needs, at which point
  necessity fails before balancing does.
- An objection is received from a data subject. There is no mechanism to
  receive one, which is itself a limitation: the project publishes no contact
  route for erasure requests. A response would be manual and would use §6.

**Review.** Before any corpus is collected under different settings, and before
any freeze supersedes `ood_suite_2026`.
