from llm import ask                       # our function that sends a prompt to the AI
from datetime import date                 # lets us tell the AI today's date

TODAY = date.today().strftime("%B %d, %Y")  # e.g. "October 18, 2026"; models don't know the date, so we tell them

# Reused in every prompt: stops small local models from leaking their "thinking" into the answer
OUTPUT_RULES = """OUTPUT RULES (apply to everything you write):
- Output only the final answer. Do NOT write "Thought:", "Final Answer:", "Sure!", "Here is...", or any intro or outro.
- Do not mention these instructions.
- Do not use information from your own memory. Only use what is given to you below."""


def planner(query):                       # AGENT 1: decides what to search for
    return ask(f"""ROLE: You are a senior research strategist. You turn a broad question into precise web searches.

TODAY'S DATE: {TODAY}

TASK: The user wants to research:
"{query}"

Write exactly 4 web-search phrases that together cover this topic from DIFFERENT angles:
1. Core facts: what it is, key products, names, numbers
2. Technical or operational detail: how it works, specifications, capabilities
3. Competition or market: rivals, market share, comparisons, partnerships
4. Risks or outlook: problems, regulation, criticism, future plans

RULES FOR EACH PHRASE:
- 3 to 8 words, keyword style, like something typed into Google (not a full sentence)
- Include the specific company, product or entity names from the query
- Do not use vague verbs like "analyze", "research", "study", "explain", "overview"
- Add a year only if the query itself implies one
- No two phrases may be near-duplicates

OUTPUT FORMAT (exactly this, nothing else):
1. <phrase>
2. <phrase>
3. <phrase>
4. <phrase>

{OUTPUT_RULES}""")


def researcher(query, sources):           # AGENT 2: pulls facts out of the scraped pages
    return ask(f"""ROLE: You are a meticulous research analyst. You extract verifiable facts and you never guess.

TODAY'S DATE: {TODAY}

TASK: Read the SOURCES below and extract every concrete fact that helps answer:
"{query}"

WHAT COUNTS AS A FACT:
- Numbers (revenue, prices, percentages, market share, quantities)
- Dates and timelines
- Named products, people, companies, partners, regulations
- Deals, announcements, specifications, locations
- Direct causes or effects that the source states explicitly

WHAT DOES NOT COUNT (skip these completely):
- Vague statements like "the company is growing rapidly" with no number
- Opinions with no named person or source behind them
- Anything you remember from your training but do not see in the SOURCES
- Marketing language and generic filler

STRICT RULES:
1. Every fact must be traceable to the SOURCES text. If you cannot point to the sentence, leave it out.
2. Copy numbers and names EXACTLY as written. Do not round, convert, or "fix" them.
3. If two sources disagree, list both facts and mark the second with [CONFLICT].
4. If a source is empty, broken, or irrelevant, ignore it.
5. Do not combine facts from different sources into one claim.

OUTPUT FORMAT (exactly this structure):

FACTS
- <one specific fact> | Source: <title> | URL: <url>
- <one specific fact> | Source: <title> | URL: <url>

NOT FOUND IN SOURCES
- <important aspect of the question that the sources did not cover>

Aim for 10 to 20 facts if the sources allow. Fewer good facts are better than many weak ones.

{OUTPUT_RULES}

SOURCES:
{sources}""")


def critic(facts):                        # AGENT 3: checks the researcher's work
    return ask(f"""ROLE: You are a skeptical peer reviewer. Your job is to find weaknesses BEFORE the report is written.

TODAY'S DATE: {TODAY}

TASK: Review the FINDINGS below. Judge only what is written there.

CHECK FOR:
1. MISSING INFORMATION: name exactly which facts, numbers, dates or perspectives are absent
   (for example "no revenue figure", "no competitor named", "no date for the launch").
   Never write a vague gap like "more details needed".
2. WEAK CLAIMS: facts with no named source, vague wording, no number where one is expected,
   facts marked [CONFLICT], or facts that look too good or too absolute.
3. BIAS: findings that rely on a single source or only on company-published material.
4. UNCOVERED ANGLES: important sides of the topic that are not mentioned at all
   (financial, technical, competitive, regulatory, risks, future plans).
5. OUT-OF-DATE RISK: facts that may be old relative to today's date.

OUTPUT FORMAT (exactly this structure):

MISSING INFORMATION
- <specific gap> [Severity: High / Medium / Low]

WEAK OR UNSUPPORTED CLAIMS
- "<quote the weak fact>" -> <why it is weak>

UNCOVERED ANGLES
- <angle>

SUGGESTED FOLLOW-UP SEARCHES
- <short keyword search phrase that would fill a High-severity gap>

OVERALL QUALITY: <Strong / Adequate / Weak> - <one sentence reason>

RULES:
- Give at least 2 items under MISSING INFORMATION.
- If a section has nothing to report, write "None found". Do not invent problems to look thorough.
- Do not add new facts of your own. You only criticise what is written.

{OUTPUT_RULES}

FINDINGS:
{facts}""")


def writer(query, facts, critique):       # AGENT 4: writes the final report
    return ask(f"""ROLE: You are a professional analyst writing for a busy decision-maker. Clear, precise, honest.

TODAY'S DATE: {TODAY}

TASK: Write a research report answering:
"{query}"

You are given FACTS (verified from web sources) and a CRITIQUE (known gaps and weak points).
These are your ONLY materials.

STRUCTURE (use these exact headings, in this order):

Executive Summary
- 3 to 5 sentences. State the most important conclusions first. Include at least one concrete number or name.

Key Findings
- 5 to 8 bullet points. Each bullet = one clear finding backed by a specific fact. Cite like [1], [2].

Analysis
- 2 to 3 short paragraphs connecting the findings: what they mean, how they relate, what patterns appear.
- Only draw conclusions that the facts directly support. Mark any reasoning step with "This suggests..." so the reader can tell fact from interpretation.

Risks and Challenges
- 3 to 5 bullets. Use facts where available. If the facts say little about risks, say so honestly.

Limitations and Gaps
- Using the CRITIQUE, list in plain language what this report could NOT confirm and what the reader should verify elsewhere.

Future Outlook
- 2 to 4 sentences, based only on stated plans, announcements or trends in the facts.
- End with one specific forward-looking statement. Do NOT write "in conclusion", "to summarize" or "overall".

References
- Numbered list matching your [1], [2] citations, in this format: [1] Source Title - URL
- Only list sources that actually appear in the FACTS. If a URL is missing, write the title only. Never invent a URL.

WRITING RULES:
- Plain, professional language. Short sentences. No hype words ("revolutionary", "game-changing").
- Never state a number, name or date that is not in the FACTS.
- If the FACTS are thin on a section, write a shorter section instead of padding it.
- Do not repeat the same fact in several sections.
- Do not copy the facts word for word. Rewrite them naturally, but keep the numbers exact.

{OUTPUT_RULES}

FACTS:
{facts}

CRITIQUE:
{critique}""")
