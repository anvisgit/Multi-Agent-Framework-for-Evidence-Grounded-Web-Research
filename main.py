from web import get_sources
from agents import planner, researcher, critic, writer


def parse_plan(plan_text):
    phrases = []
    for line in plan_text.splitlines():
        line = line.strip()
        if line and line[0].isdigit():
            phrases.append(line.lstrip("0123456789.) ").strip())
    return phrases


query = input("Research question: ").strip()

if len(query.split()) < 4:
    print("Too vague. Be more specific, e.g. 'NVIDIA Blackwell GPU data center 2024'")
    raise SystemExit

print("\n[1/4] Planning...")
plan = planner(query)
print(plan)

print("\n[2/4] Searching the web and extracting facts...")
all_sources = get_sources(query, n=3)
for phrase in parse_plan(plan)[:4]:
    print("  searching:", phrase)
    all_sources += "\n\n---\n\n" + get_sources(phrase, n=1)

facts = researcher(query, all_sources)
print("\nFACTS:\n", facts)

print("\n[3/4] Critiquing...")
critique = critic(facts)
print(critique)

print("\n[4/4] Writing report...")
report = writer(query, facts, critique)

print("\n" + "=" * 60)
print(report)

with open("report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("\nSaved to report.txt")