# How to object, and what happens when you do

This project holds an evaluation corpus of published news articles
(`ood_suite_2026_v4`, 266 documents from 41 publishers, all published after
2026-01-01). News text names people, so if you are named or quoted in one of
those articles, this project processes personal data about you, and you have
rights over that processing.

The lawful basis is legitimate interest for research, assessed in
[`docs/balancing-test.md`](balancing-test.md). That assessment concluded the
interest carries **narrowly**, and it named the absence of a route to object as
one of the things weakening its own reasoning. This document is that route.

## Contact

> **Outstanding.** A dedicated address is being created for this. Until the
> address exists and appears on this line, there is no working contact for this
> project, and that is stated here rather than papered over with a placeholder.
> The crawler's user agent currently advertises `https://example.invalid/bot`,
> which is a reserved domain that can never resolve, so no publisher contacted
> during collection had a working route either. Both are filled from the same
> address when it exists.

## What you can ask for

**To object to the processing** (GDPR Article 21). Because the basis is
legitimate interest, you can object at any time and the processing must stop
unless compelling legitimate grounds override your interests. For a corpus this
small, held for one evaluation and never redistributed, the realistic answer is
that it stops.

**Erasure** (Article 17). Removal of the documents that concern you.

**Access** (Article 15). What is held, where it came from, and what it is used
for. For this corpus that is: the article text as published, the URL it came
from, its publication date, the publisher's name, and derived counts. No
profile, no inference about you, and no record linking you across documents —
the corpus is indexed by article and by publisher, never by person.

**Rectification** (Article 16) is available but will usually be the wrong
route. This corpus holds articles as the publisher published them. If a fact
about you in an article is wrong, the correction belongs with the publisher; a
corrected copy here would misrepresent what was published, which is the thing
the corpus records.

## What happens to a request

**1. Received and acknowledged.** Within 5 working days, in writing, saying
what will happen and by when.

**2. Identity checked, proportionately.** Only enough to be confident the
request concerns the person making it — normally the name as it appears in the
article, and one detail linking you to it. Article 12(6) permits asking for more
where there is genuine doubt. It does not permit collecting an identity document
for a request about a published newspaper article, and none will be asked for.
Anything supplied for verification is deleted once the request is closed, and is
never added to the corpus.

**3. Located.** Every document is searched for the name and for obvious
variants. **This is a text search, and it is the honest limit of the
procedure**: the corpus has no index by person, deliberately — the balancing
test relies on that as a safeguard. So a name that appears only as an
unattributed description, or is spelled differently, may not be found. If you
know which article or publisher is involved, saying so makes the search
reliable rather than best-effort.

**4. Decided and actioned.** Within one month of receipt, extendable by two
further months for complex requests, with reasons given before the first month
ends (Article 12(3)).

Erasure or a sustained objection is actioned by rebuilding the frozen corpus
without the affected documents. The suite does not change in place — a frozen
corpus is superseded, never edited, so the result is a new freeze with a new
hash and a new manifest entry, and the old one is deleted with a receipt:

```bash
PYTHONPATH=. .venv/bin/python -m eval.ood_collection.erase data/ood/frozen-v4
```

That prints what would go and deletes nothing. Adding `--yes` deletes and
appends a receipt naming every file removed with its SHA-256, so the deletion
can be checked afterwards rather than believed.

Any analysis already published from the superseded corpus keeps its original
numbers and is annotated to say the corpus changed and why — in general terms,
never naming the person who asked.

**5. Logged.** In a register holding: the date received, the category of
request, the decision, the action taken, and the date closed. **The register
records no more personal data than the request itself required, and identity
evidence is not kept.** A compliance log that quietly becomes a second personal
data store is a common failure and is avoided here by writing down only what a
later auditor needs to confirm the request was handled.

## If you are not satisfied

You can complain to a supervisory authority. If you are in the EU, that is the
data protection authority of your member state, and you may also complain to the
authority where this project's controller is established.

## What this project will not do

It will not ask you to prove identity beyond what the request needs. It will not
retain what you send beyond closing the request. It will not add you to any list
in order to remember that you objected — the only record is the register entry
above.

And it will not make any assessment of whether an article about you is true.
The system does not output a truth verdict about any claim or any person
(D001), so an objection here removes text from an evaluation corpus. It does not
adjudicate anything about you, and nothing in this project ever will.
