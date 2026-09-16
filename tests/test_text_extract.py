"""Publisher marker stripping: the OOD construction rule from DATA.md.

These are not tidying assertions. The dateline prefix is the single strongest
publisher tell in the corpus the leakage analysis criticises, and letting one
through into the OOD suite would reintroduce the leakage this project exists
to document. Each case is a form real wire copy actually uses.
"""

from __future__ import annotations

import pytest
from pipeline.text.extract import (
    extract_article,
    mask_self_links,
    strip_publisher_markers,
)

DATELINES = [
    "LONDON (Reuters) - Body follows.",
    "WASHINGTON (AP) - Body follows.",
    "NEW YORK, Jan 5 (Reuters) - Body follows.",
    "PARIS, France (CNN) -- Body follows.",
    "SAN FRANCISCO (Bloomberg) - Body follows.",
    "BRUSSELS (AFP) - Body follows.",
]

BYLINES = [
    "By Jane Doe\nBody follows.",
    "By Jane Doe, Staff Writer\nBody follows.",
    "Reporting by Li Wei and Ahmed Khan\nBody follows.",
    "Written by Maria Garcia\nBody follows.",
]


@pytest.mark.parametrize("raw", DATELINES)
def test_datelines_are_removed(raw: str) -> None:
    out, removed = strip_publisher_markers(raw)
    assert out.strip() == "Body follows."
    assert removed, "removal must be recorded, not silent"


@pytest.mark.parametrize("raw", BYLINES)
def test_bylines_are_removed(raw: str) -> None:
    out, removed = strip_publisher_markers(raw)
    assert out.strip() == "Body follows."
    assert "byline" in removed


def test_em_and_en_dash_datelines_both_match() -> None:
    """Wire copy uses typographic dashes; ASCII-only regexes miss most of it."""
    for dash in ("-", chr(0x2013), chr(0x2014)):
        out, removed = strip_publisher_markers(f"LONDON (Reuters) {dash} Body follows.")
        assert out.strip() == "Body follows.", f"dash {hex(ord(dash))} not handled"
        assert removed


def test_ordinary_sentence_is_not_mistaken_for_a_dateline() -> None:
    """The stripper must not eat real content.

    A false positive here silently deletes the first sentence of an article,
    which would be a worse corpus defect than the one being removed.
    """
    for keeper in (
        "The FBI (Federal Bureau of Investigation) said the report was wrong.",
        "NASA said the launch would proceed.",
        "In 2024 the committee - which had met twice - reported back.",
    ):
        out, removed = strip_publisher_markers(keeper)
        assert out.strip() == keeper, f"content destroyed: {out!r}"
        assert not removed


def test_removal_counts_are_reported_per_document() -> None:
    """Prevalence per stratum has to be reportable, not assumed."""
    _, removed = strip_publisher_markers(
        "By Jane Doe\nLONDON (Reuters) - One.\nCopyright 2026 Example Corp."
    )
    assert removed.get("byline") == 1
    assert removed.get("dateline") == 1
    assert removed.get("boilerplate") == 1


def test_end_to_end_extraction_strips_what_trafilatura_leaves() -> None:
    """Regression guard for the measured trafilatura gap.

    trafilatura alone returns the byline and dateline inside the body text.
    If this test ever fails open, the OOD suite is being built with publisher
    identity intact.
    """
    # Distinct paragraphs, long enough to clear the article length floor, so
    # this exercises the stripping rather than the gate. They must differ:
    # trafilatura deduplicates identical blocks, which collapsed an earlier
    # repeated fixture below the floor.
    filler = "".join(
        f"<p>{sentence}</p>"
        for sentence in (
            "Officials said the vote had been expected for several weeks.",
            "Consultation with residents closed at the end of last month.",
            "Local business groups had asked for a longer transition period.",
            "The finance committee published its recommendations in June.",
            "Two amendments were withdrawn before the final reading began.",
            "Spending on transport rises by four percent under the plan.",
            "Opposition members requested a recorded division on the motion.",
            "The measures take effect at the start of the next financial year.",
        )
    )
    html = (
        "<html><body><nav>Subscribe</nav><article>"
        "<h1>Council approves budget</h1>"
        "<p>By Jane Doe, Staff Writer</p>"
        "<p>LONDON (Reuters) - The council approved the budget on Tuesday.</p>"
        f"{filler}"
        "<p>Our Standards: The Thomson Reuters Trust Principles.</p>"
        "</article><footer>Copyright 2026</footer></body></html>"
    )
    article = extract_article(html)
    assert article is not None
    assert "Reuters" not in article.text, "agency name survived into the corpus"
    assert "Jane Doe" not in article.text, "byline survived into the corpus"
    assert "Subscribe" not in article.text
    assert "council approved the budget" in article.text


def test_navigation_only_page_is_not_an_article() -> None:
    """Measured: trafilatura returns "Menu" for a nav-only page.

    Without a length floor that fragment enters the corpus as a document, so
    the floor is a corpus construction control, not a nicety.
    """
    assert extract_article("<html><body><nav>Menu</nav></body></html>") is None


def test_short_fragment_is_rejected_but_real_article_survives() -> None:
    body = "The council approved the budget on Tuesday evening. " * 12
    html = f"<html><body><article><h1>Budget</h1><p>{body}</p></article></body></html>"
    article = extract_article(html)
    assert article is not None, "a full length article must survive the floor"
    assert len(article.text) >= 400


CAPS_BYLINES = [
    "DAVID MARCUS: Congress set to protect AM radio.",
    "JANE DOE: The council met on Tuesday.",
    "MARY-ANN O'BRIEN: Spending rose again.",
]

NOT_BYLINES = [
    "BREAKING: The council met on Tuesday.",
    "UPDATE: Spending rose again.",
    "The council met: it approved the budget.",
    "NASA: the launch will proceed.",
]


@pytest.mark.parametrize("raw", CAPS_BYLINES)
def test_caps_name_prefix_byline_is_removed(raw: str) -> None:
    """Measured on live Fox News copy: the "By" rule left these in place."""
    out, removed = strip_publisher_markers(raw)
    assert not out.strip().startswith(("DAVID", "JANE", "MARY")), out
    assert "caps_byline" in removed


@pytest.mark.parametrize("raw", NOT_BYLINES)
def test_single_word_prefixes_are_not_treated_as_bylines(raw: str) -> None:
    """A false positive here deletes the opening of a real article."""
    out, removed = strip_publisher_markers(raw)
    assert out.strip() == raw, f"content destroyed: {out!r}"
    assert "caps_byline" not in removed


# ------------------------------------------------ outlet names and promotions

from pipeline.text.extract import OUTLET_MASK, mask_outlet_names  # noqa: E402


def test_own_outlet_name_is_masked() -> None:
    text, n = mask_outlet_names(
        "Fox News Digital reached out for comment. Fox News reported it first.",
        ["Fox News Digital", "Fox News"],
    )
    assert "Fox" not in text
    assert text.count(OUTLET_MASK) == 2
    assert n == 2


def test_longest_name_is_masked_whole() -> None:
    """Otherwise "Fox News Digital" leaves "Digital" behind."""
    text, _ = mask_outlet_names("Fox News Digital said", ["Fox News", "Fox News Digital"])
    assert text == f"{OUTLET_MASK} said"


def test_masking_is_case_sensitive_and_whole_word() -> None:
    """Narrow names exist so ordinary words survive."""
    text, n = mask_outlet_names(
        "The reason is simple: voxel art and a vox pop on Capitol Hill.",
        ["Reason magazine", "Vox", "The Hill"],
    )
    assert n == 0
    assert "reason" in text and "voxel" in text and "vox pop" in text


def test_possessive_is_masked() -> None:
    text, n = mask_outlet_names("Vox's reporting found", ["Vox"])
    assert text == f"{OUTLET_MASK}'s reporting found"
    assert n == 1


def test_domain_form_is_masked_without_eating_neighbours() -> None:
    text, _ = mask_outlet_names("Read more at Reason.com today.", ["Reason.com"])
    assert text == f"Read more at {OUTLET_MASK} today."


def test_mid_paragraph_newsletter_promo_is_removed() -> None:
    out, removed = strip_publisher_markers(
        "The vote passed. This story was first featured in the Future Perfect "
        "newsletter. Sign up here to get it. Officials said more would follow."
    )
    assert "newsletter" not in out
    assert "The vote passed." in out
    assert "Officials said more would follow." in out
    assert removed.get("newsletter_promo", 0) >= 1


def test_subscribe_without_newsletter_is_content() -> None:
    raw = "Members were urged to subscribe to the new pension scheme before March."
    out, removed = strip_publisher_markers(raw)
    assert out.strip() == raw
    assert "newsletter_promo" not in removed


def test_outlet_masking_counts_are_reported() -> None:
    _, removed = strip_publisher_markers("The Guardian understands this.", ["The Guardian"])
    assert removed["outlet_name"] == 1


def test_invisible_flag_reads_extracted_text_not_raw_html() -> None:
    """Raw pages carry zero width characters in scripts; that is not injection."""
    zwj = chr(0x200D)
    body = " ".join(f"Sentence {i} of a perfectly ordinary article body." for i in range(16))
    html = (
        f"<html><head><script>var s='{zwj}';</script></head>"
        f"<body><article><h1>Head</h1><p>{body}</p></article></body></html>"
    )
    article = extract_article(html)
    assert article is not None
    assert article.had_invisible is False


# ------------------------------------------------------------------ language

from pipeline.text.extract import declared_language, english_ratio  # noqa: E402

SPANISH = (
    "El consejo dijo que votaría el presupuesto en marzo y los funcionarios no se "
    "sorprendieron por la decisión tomada durante la sesión extraordinaria del lunes."
)
ENGLISH = (
    "The council said that it would vote on the budget in March, and officials were "
    "not surprised by the decision that was taken at the special session on Monday."
)


def test_declared_language_is_read_from_the_html_element() -> None:
    assert declared_language('<html class="x" lang="es-419"><body>') == "es-419"
    assert declared_language("<html><body>") is None


def test_english_ratio_separates_english_from_spanish() -> None:
    assert english_ratio(ENGLISH) > 0.3
    assert english_ratio(SPANISH) < 0.1


def test_numbers_do_not_read_as_foreign() -> None:
    assert english_ratio("") == 0.0
    assert english_ratio("The rate was 4.5 percent in 2026, and it is rising.") > 0.3


def test_extracted_article_carries_language_signals() -> None:
    body = " ".join(ENGLISH for _ in range(6))
    article = extract_article(
        f'<html lang="en-GB"><body><article><p>{body}</p></article></body></html>'
    )
    assert article is not None
    assert article.declared_language == "en-GB"
    assert article.english_ratio > 0.3


# --------------------------------------------- widgets and self disclosures


def test_named_newsletter_widget_line_is_removed() -> None:
    """Measured in pilot 2: the brand and a staff name rode in on this line."""
    raw = (
        "The committee met on Tuesday.\n"
        "The Reason Roundup Newsletter by Liz Wolfe Liz and Reason help you make sense "
        "of the day's news every morning.\n"
        "Officials said more would follow."
    )
    out, removed = strip_publisher_markers(raw)
    assert "Roundup" not in out and "Liz Wolfe" not in out
    assert "The committee met on Tuesday." in out
    assert "Officials said more would follow." in out
    assert removed["newsletter_promo"] == 1


def test_named_newsletter_in_prose_is_content() -> None:
    raw = "Officials cited the Parish Council Newsletter as the source of the figures."
    out, removed = strip_publisher_markers(raw)
    assert out.strip() == raw
    assert "newsletter_promo" not in removed


def test_self_disclosure_and_its_denial_are_removed() -> None:
    raw = (
        "Officials met. The Center for Investigative Reporting, the parent company of "
        "Mother Jones, has sued OpenAI for copyright violations. OpenAI denies the "
        "allegations. More followed."
    )
    out, removed = strip_publisher_markers(raw, ["Mother Jones"])
    assert "parent company" not in out and "OpenAI" not in out
    assert "Officials met." in out and "More followed." in out
    assert removed["self_disclosure"] == 2


def test_ownership_about_another_company_is_content() -> None:
    raw = "Amazon is owned by its shareholders, the filing said."
    out, removed = strip_publisher_markers(raw, ["Mother Jones"])
    assert out.strip() == raw
    assert "self_disclosure" not in removed


# -------------------------------------------- contact, follow and credit lines

from pipeline.text.extract import EMAIL_MASK  # noqa: E402


def test_staff_bio_with_contact_details_is_removed() -> None:
    """Measured on Breitbart in the full run, including a reporter's email."""
    raw = (
        "The vote passed. Joel Klein is a reporter for Breitbart News. Email him at "
        "jklein@example.com. Follow him on Twitter @JoelKlein."
    )
    out, removed = strip_publisher_markers(raw, ["Breitbart News", "Breitbart"])
    assert out.strip() == "The vote passed."
    assert "jklein" not in out and "@JoelKlein" not in out
    assert removed["email_address"] == 1
    assert removed["staff_bio"] == 1
    assert removed["follow_or_contact"] == 2


def test_follow_us_promo_is_removed() -> None:
    raw = "Get the latest money news, and follow us on TikTok, Facebook and X. Officials spoke."
    out, removed = strip_publisher_markers(raw)
    assert out.strip() == "Officials spoke."
    assert removed["follow_or_contact"] == 1


def test_following_as_ordinary_verb_is_content() -> None:
    raw = "Investors will follow the central bank closely this week."
    out, removed = strip_publisher_markers(raw)
    assert out.strip() == raw
    assert "follow_or_contact" not in removed


def test_signing_up_form_of_promo_is_removed() -> None:
    """Liverpool Echo in the full run; the rule only knew "sign up"."""
    raw = "Stay updated by signing up to our daily and breaking newsletter. The match ended."
    out, removed = strip_publisher_markers(raw)
    assert out.strip() == "The match ended."
    assert removed["newsletter_promo"] >= 1


def test_contributor_credit_is_removed() -> None:
    raw = "The book tour continues. Rafaela Jinich contributed to this newsletter."
    out, removed = strip_publisher_markers(raw)
    assert out.strip() == "The book tour continues."
    assert removed["contributor_credit"] == 1


def test_parenthetical_photo_credit_is_removed() -> None:
    raw = "Veterans gathered (Photo by Craig F. Walker/Example via Getty Images) in the square."
    out, removed = strip_publisher_markers(raw)
    assert "Walker" not in out
    assert out.strip() == "Veterans gathered in the square."
    assert removed["photo_credit"] == 1


def test_email_in_reporting_is_masked_not_dropped() -> None:
    out, removed = strip_publisher_markers("Tips can be sent to tips@example.org by Friday.")
    assert out.strip() == f"Tips can be sent to {EMAIL_MASK} by Friday."
    assert removed["email_address"] == 1


# ----------------------------------- self promotion, affiliate and membership


def test_reach_directly_form_of_contact_is_removed() -> None:
    out, removed = strip_publisher_markers("The vote passed. Reach him directly at the desk.")
    assert out.strip() == "The vote passed."
    assert removed["follow_or_contact"] == 1


def test_self_promotion_keyed_on_the_mask_is_removed() -> None:
    raw = "The range day went ahead. You can sign up to get Down Range at Breitbart News."
    out, removed = strip_publisher_markers(raw, ["Breitbart News"])
    assert out.strip() == "The range day went ahead."
    assert removed["self_promo"] == 1


def test_sign_up_without_the_outlet_is_content() -> None:
    raw = "Residents can sign up for the council's flood alerts online."
    out, removed = strip_publisher_markers(raw)
    assert out.strip() == raw
    assert "self_promo" not in removed


def test_affiliate_disclosure_and_support_line_are_removed() -> None:
    raw = (
        "The novel is long. When you buy a book using a link in this newsletter, we "
        "receive a commission. Thank you for supporting The Atlantic."
    )
    out, removed = strip_publisher_markers(raw, ["The Atlantic"])
    assert out.strip() == "The novel is long."
    assert removed["affiliate_disclosure"] == 1
    assert removed["self_promo"] == 1


def test_membership_and_sponsor_lines_are_removed() -> None:
    raw = (
        "Rates held steady. Get the U.S. Chamber's free newsletter for insights on the economy. "
        "Members get access to all of our articles, members-only newsletters, and podcasts."
    )
    out, removed = strip_publisher_markers(raw)
    assert out.strip() == "Rates held steady."
    assert removed["membership_or_sponsor"] == 2


def test_ordinary_uses_of_get_and_members_are_content() -> None:
    raw = "Get the facts before you vote. The union said members only voted once."
    out, removed = strip_publisher_markers(raw)
    assert out.strip() == raw
    assert "membership_or_sponsor" not in removed


def test_domain_inside_an_email_address_is_not_masked() -> None:
    """The address is handled as personal data, not mangled into a half mask."""
    text, n = mask_outlet_names("write to jklein@breitbart.com today", ["breitbart.com"])
    assert n == 0
    assert text == "write to jklein@breitbart.com today"


# --- rules added after the spread run audit, 2026-09-15 ---------------------


def test_institutional_byline_is_removed() -> None:
    """No surname to match, so the "By Name Name" rule never saw it."""
    text = "By University of Oregon School of Journalism and Communication\nThe report says."
    out, counts = strip_publisher_markers(text)
    assert "University of Oregon" not in out
    assert counts["byline"] == 1


def test_a_sentence_opening_with_a_weekday_is_not_a_byline() -> None:
    """The rule above must not eat prose."""
    text = "By Friday afternoon, Hezbollah had issued a statement.\nMore followed."
    out, counts = strip_publisher_markers(text)
    assert "Hezbollah had issued a statement" in out
    assert "byline" not in counts


def test_a_byline_with_a_long_role_is_removed() -> None:
    text = "By Aishwarya Balamukundan, Senior Director of Research and Advisory\nText follows."
    out, counts = strip_publisher_markers(text)
    assert "Aishwarya" not in out
    assert counts["byline"] == 1


def test_photo_credit_as_its_own_sentence_is_removed() -> None:
    text = "The order stands. Illustration by Noah Hickey. The piece continues."
    out, counts = strip_publisher_markers(text)
    assert "Noah Hickey" not in out
    assert counts["photo_credit"] == 1


def test_illustration_in_prose_survives() -> None:
    text = "Illustration by the artist shows a ship in the strait."
    out, _ = strip_publisher_markers(text)
    assert "artist shows a ship" in out


def test_a_permission_notice_inside_a_sentence_is_removed() -> None:
    """Line anchored boilerplate never saw this: it sits mid paragraph."""
    text = "The chart is used herein with permission. All rights reserved. Growth slowed."
    out, counts = strip_publisher_markers(text)
    assert "all rights reserved" not in out.lower()
    assert "Growth slowed" in out
    assert counts["permission_notice"] >= 1


def test_self_links_are_masked_including_sibling_domains() -> None:
    """Hosts are lowercase and unspaced, so the name mask never catches them."""
    text = (
        "Visit https://www.bbc.co.uk/learningenglish/newsletters for more, "
        "or bbclearningenglish.com/programmes, or https://bbc.in/newscast."
    )
    out, n = mask_self_links(text, "bbc.com")
    assert "bbc" not in out.lower()
    assert n == 3


def test_a_third_party_link_is_left_alone() -> None:
    text = "The filing is at https://www.supremecourt.gov/opinions/24pdf/23-1.pdf today."
    out, n = mask_self_links(text, "bbc.com")
    assert "supremecourt.gov" in out
    assert n == 0


def test_a_short_label_masks_nothing() -> None:
    """Below the floor the compound test would match half the web."""
    _, n = mask_self_links("See https://upside.example.com/x now.", "up.com")
    assert n == 0


def test_a_caption_label_without_a_credit_word_is_removed() -> None:
    text = "Never miss a moment. Photo: Vessel in the Strait of Hormuz. The report continues."
    out, counts = strip_publisher_markers(text)
    assert "Strait of Hormuz" not in out
    assert counts["photo_credit"] == 1


def test_a_colon_in_prose_is_not_a_caption() -> None:
    out, _ = strip_publisher_markers("He described the photo: a blurred figure on a roof.")
    assert "blurred figure" in out


def test_the_article_case_is_relaxed_when_masking_a_two_word_self_name() -> None:
    """outlet_names.yaml records "the Post"; sentences open "The Post"."""
    text = "The Post's broader goal is stated. Readers of the Post know it."
    out, n = mask_outlet_names(text, ("the Post",))
    assert "Post" not in out
    assert n == 2


def test_relaxing_the_article_does_not_relax_the_name() -> None:
    out, n = mask_outlet_names("They walked the post road at dawn.", ("the Post",))
    assert "post road" in out
    assert n == 0
