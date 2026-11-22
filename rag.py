"""
CityPulse AI - Knowledge Assistant (RAG).

Retrieval over GROUNDED sources only:
  (a) Socrata dataset metadata for v6vf-nfxy (field definitions, fetched from
      the API and cached; a verified fallback copy is used offline),
  (b) per-SR-type knowledge cards computed from the REAL cached data
      (department, median days-to-close, >7-day breach rate, volume, severity),
  (c) the project README + how-priority-works explainer,
  (d) per-community-area summary cards (volume, top types, median close time,
      current hotspot state).

Index: TF-IDF (sklearn) - dependency-light and fully offline.
Answering: if OPENAI_API_KEY or ANTHROPIC_API_KEY is set in the environment,
an LLM composes a grounded answer citing the retrieved sources. Otherwise
(the default) the answer is EXTRACTIVE: the structured card + retrieved chunks
are returned verbatim with source labels. No number is ever invented - if it
is not in the retrieved material, it is not in the answer.
"""
import json
import os
import re

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
DATA = os.path.join(HERE, "data")

AREA_NAMES = {
    1: "Rogers Park", 2: "West Ridge", 3: "Uptown", 4: "Lincoln Square",
    5: "North Center", 6: "Lake View", 7: "Lincoln Park", 8: "Near North Side",
    9: "Edison Park", 10: "Norwood Park", 11: "Jefferson Park", 12: "Forest Glen",
    13: "North Park", 14: "Albany Park", 15: "Portage Park", 16: "Irving Park",
    17: "Dunning", 18: "Montclare", 19: "Belmont Cragin", 20: "Hermosa",
    21: "Avondale", 22: "Logan Square", 23: "Humboldt Park", 24: "West Town",
    25: "Austin", 26: "West Garfield Park", 27: "East Garfield Park",
    28: "Near West Side", 29: "North Lawndale", 30: "South Lawndale",
    31: "Lower West Side", 32: "Loop", 33: "Near South Side", 34: "Armour Square",
    35: "Douglas", 36: "Oakland", 37: "Fuller Park", 38: "Grand Boulevard",
    39: "Kenwood", 40: "Washington Park", 41: "Hyde Park", 42: "Woodlawn",
    43: "South Shore", 44: "Chatham", 45: "Avalon Park", 46: "South Chicago",
    47: "Burnside", 48: "Calumet Heights", 49: "Roseland", 50: "Pullman",
    51: "South Deering", 52: "East Side", 53: "West Pullman", 54: "Riverdale",
    55: "Hegewisch", 56: "Garfield Ridge", 57: "Archer Heights", 58: "Brighton Park",
    59: "McKinley Park", 60: "Bridgeport", 61: "New City", 62: "West Elsdon",
    63: "Gage Park", 64: "Clearing", 65: "West Lawn", 66: "Chicago Lawn",
    67: "West Englewood", 68: "Englewood", 69: "Greater Grand Crossing",
    70: "Ashburn", 71: "Auburn Gresham", 72: "Beverly", 73: "Washington Heights",
    74: "Mount Greenwood", 75: "Morgan Park", 76: "O'Hare", 77: "Edgewater",
}
NAME_TO_AREA = {v.lower(): k for k, v in AREA_NAMES.items()}

FIELD_DEFS_FALLBACK = {
    "sr_number": "Unique service request number assigned by the City of Chicago 311 system.",
    "sr_type": "The type of service request (for example Pothole in Street Complaint).",
    "owner_department": "The city department that owns and works the request.",
    "status": "Request status: Open, Completed or Canceled in this dataset.",
    "created_date": "Date and time the request was created.",
    "closed_date": "Date and time the request was completed/closed.",
    "community_area": "Chicago community area number (1-77) where the request is located.",
    "ward": "Chicago ward number where the request is located.",
    "duplicate": "The City of Chicago's own flag for whether this request is a duplicate of another.",
    "street_address": "Street address of the reported problem.",
    "latitude/longitude": "Geographic coordinates of the reported problem.",
}


def _socrata_field_chunks():
    cache = os.path.join(DATA, "socrata_metadata.json")
    cols = None
    try:
        if os.path.exists(cache):
            cols = json.load(open(cache))
        else:
            r = requests.get("https://data.cityofchicago.org/api/views/v6vf-nfxy", timeout=30)
            r.raise_for_status()
            cols = r.json().get("columns", [])
            json.dump(cols, open(cache, "w"))
    except Exception:
        cols = None
    chunks = []
    if cols:
        for c in cols:
            desc = (c.get("description") or "").strip()
            if desc:
                chunks.append({"source": "Socrata dataset dictionary (v6vf-nfxy)",
                               "title": f"Field: {c.get('name')}",
                               "text": f"{c.get('name')}: {desc}"})
    if not chunks:
        for k, v in FIELD_DEFS_FALLBACK.items():
            chunks.append({"source": "Socrata dataset dictionary (v6vf-nfxy, cached fallback)",
                           "title": f"Field: {k}", "text": f"{k}: {v}"})
    return chunks


def build_kb():
    import priority as prio_mod
    chunks = []
    chunks += _socrata_field_chunks()

    # (b) per-type cards from real data
    ts = pd.read_csv(os.path.join(OUT, "type_stats.csv"))
    closed_p = os.path.join(DATA, "closed_requests.parquet")
    closed = (pd.read_parquet(closed_p) if os.path.exists(closed_p)
              else pd.read_csv(os.path.join(DATA, "closed_requests.csv"), low_memory=False))
    dept_mode = closed.groupby("sr_type")["owner_department"].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else "Unknown")
    for _, r in ts.iterrows():
        tier, score = prio_mod.severity_for(r["sr_type"])
        chunks.append({
            "source": "Computed from real Chicago 311 data (type_stats)",
            "title": f"Request type: {r['sr_type']}",
            "text": (f"{r['sr_type']}: handled by {dept_mode.get(r['sr_type'], 'Unknown')}. "
                     f"Based on {int(r['n']):,} completed requests in the cached data, the "
                     f"median time to close is {r['median_days']:.1f} days and "
                     f"{r['breach_rate']*100:.0f}% took more than 7 days. "
                     f"CityPulse severity tier: {tier} (score {score}) - "
                     f"{'safety or essential-service impact' if score >= 0.75 else 'service or cosmetic impact'} "
                     f"under the hand-set severity policy in priority.py.")})

    # (d) per-area cards
    fair = pd.read_csv(os.path.join(OUT, "fairness.csv"))
    hot = pd.read_csv(os.path.join(OUT, "hotspots.csv"))
    ah = pd.read_csv(os.path.join(OUT, "area_hotspots.csv"))
    ah_idx = {str(r.community_area): r for r in ah.itertuples()}
    for _, r in fair.iterrows():
        area = str(r["community_area"])
        try:
            name = AREA_NAMES.get(int(float(area)), f"Area {area}")
        except (TypeError, ValueError):
            continue
        top_types = (hot[hot["community_area"].astype(str) == area]
                     .sort_values("count_7d", ascending=False).head(3))
        top_txt = ", ".join(f"{t.sr_type} ({int(t.count_7d)} this week)" for t in top_types.itertuples())
        a = ah_idx.get(area)
        surge = (f"Overall volume is running at {a.ratio:.1f}x the normal level this week."
                 if a is not None and pd.notna(a.ratio) else "")
        chunks.append({
            "source": "Computed from real Chicago 311 data (area summary)",
            "title": f"Community area {area}: {name}",
            "text": (f"Community area {area} ({name}): {int(r['open_n']):,} open requests in the "
                     f"current queue, mean priority {r['mean_priority']:.2f}. Historically the "
                     f"median time to close here is {r['median_days']:.1f} days and "
                     f"{r['breach_rate']*100:.0f}% of requests took more than 7 days. "
                     f"Most reported this week: {top_txt}. {surge}")})

    # hotspot snapshot chunks
    flagged = hot[hot["is_hotspot"] == True].sort_values("z_score", ascending=False).head(20)
    for _, r in flagged.iterrows():
        try:
            name = AREA_NAMES.get(int(float(r["community_area"])), "")
        except (TypeError, ValueError):
            name = ""
        chunks.append({
            "source": "Computed hotspots (last 7 days vs prior 28-day average)",
            "title": f"Hotspot: area {r['community_area']} {r['sr_type']}",
            "text": (f"Emerging hotspot: {r['sr_type']} in community area {r['community_area']} "
                     f"({name}) - {int(r['count_7d'])} requests in the last 7 days vs "
                     f"{r['expected_7d']:.0f} expected, {r['ratio']:.1f}x normal (z={r['z_score']:.1f}).")})

    # (c) README + priority explainer
    readme_p = os.path.join(HERE, "README.md")
    if os.path.exists(readme_p):
        text = open(readme_p).read()
        for i in range(0, len(text), 1200):
            chunks.append({"source": "Project README", "title": f"README chunk {i//1200 + 1}",
                           "text": text[i:i + 1200]})
    import priority as prio_mod
    chunks.append({"source": "How priority works (priority.py)",
                   "title": "How the CityPulse priority score works",
                   "text": (prio_mod.__doc__ or "") +
                           f" Balanced weights: {prio_mod.PRIORITY_MODES['Balanced']}. "
                           f"FIFO-first weights: {prio_mod.PRIORITY_MODES['FIFO-first']}. "
                           f"Urgency-first weights: {prio_mod.PRIORITY_MODES['Urgency-first']}."})
    return chunks


class KnowledgeAssistant:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks = build_kb()
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=8000)
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=4):
        from sklearn.metrics.pairwise import cosine_similarity
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        idx = sims.argsort()[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in idx if sims[i] > 0]

    # ---------------------------------------------------------- structured
    def _request_lookup(self, question):
        m = re.search(r"\b(SR\d{2}-\d+|CP-2026-\d+)\b", question.upper())
        if not m:
            return None
        sr = m.group(1)
        for name in ("prioritized_open.csv",):
            df = pd.read_csv(os.path.join(OUT, name))
            hit = df[df["sr_number"] == sr]
            if len(hit):
                r = hit.iloc[0]
                return (f"Request {sr} ({r['sr_type']} at {r['street_address']}) was ranked with "
                        f"priority score {r['priority_score']:.2f} ({r['risk_band']}). Breakdown: "
                        f"delay-risk probability {r['breach_prob']:.2f}, severity tier "
                        f"{r['severity_tier']} ({r['severity_score']}), waiting {r['wait_days']:.0f} days, "
                        f"hotspot ratio {r['hotspot_ratio']:.1f}x, {int(r['dup_group_size'])} report(s) "
                        f"at this address. Reason recorded: {r['reason']}",
                        "Computed priority queue (prioritized_open.csv)")
        try:
            import citizen
            reps = citizen.load_reports()
            hit = reps[reps["sr_number"] == sr]
            if len(hit):
                r = hit.iloc[-1]
                return (f"Citizen report {sr} ({r['sr_type']} at {r['street_address']}) has priority "
                        f"score {r['priority_score']} ({r['risk_band']}). Reason: {r['reason']}",
                        "Citizen reports store")
        except Exception:
            pass
        return None

    def answer(self, question):
        structured = self._request_lookup(question)
        hits = self.retrieve(question)
        sources = [f"{c['source']} — {c['title']}" for c, _ in hits]
        llm_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if llm_key:
            try:
                text = self._llm_answer(question, hits, structured)
                if structured:
                    text = structured[0] + "\n\n" + text
                    sources = [structured[1]] + sources
                return text, sources, "LLM-grounded (API key present)"
            except Exception:
                pass  # fall through to extractive
        parts = []
        if structured:
            parts.append(structured[0])
            sources = [structured[1]] + sources
        if hits:
            parts.append("From the knowledge base:")
            for c, s in hits[:3]:
                parts.append(f"• {c['text']}")
        if not parts:
            parts.append("I could not find grounded information for that in the CityPulse "
                         "knowledge base. Try asking about a request type, a community "
                         "area, a field definition, or a specific request ID.")
        return "\n\n".join(parts), sources, "Extractive (no LLM key)"


    def _llm_answer(self, question, hits, structured):
        context = "\n\n".join(f"[{c['source']}] {c['text']}" for c, _ in hits)
        prompt = (f"Answer using ONLY this context. Cite source names. If the context "
                  f"does not contain the answer, say so.\n\nContext:\n{context}\n\n"
                  f"Question: {question}")
        if os.environ.get("OPENAI_API_KEY"):
            r = requests.post("https://api.openai.com/v1/chat/completions",
                              headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                              json={"model": "gpt-4o-mini",
                                    "messages": [{"role": "user", "content": prompt}]}, timeout=60)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        r = requests.post("https://api.anthropic.com/v1/messages",
                          headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                                   "anthropic-version": "2023-06-01"},
                          json={"model": "claude-3-5-haiku-latest", "max_tokens": 600,
                                "messages": [{"role": "user", "content": prompt}]}, timeout=60)
        r.raise_for_status()
        return r.json()["content"][0]["text"]


_instance = None


def assistant():
    global _instance
    if _instance is None:
        _instance = KnowledgeAssistant()
    return _instance


if __name__ == "__main__":
    ka = assistant()
    for q in ["How long does a pothole complaint usually take to close?",
              "Which department handles street light outages?",
              "What is happening in Logan Square this week?",
              "How does the priority score work?"]:
        ans, srcs, mode = ka.answer(q)
        print("=" * 80)
        print("Q:", q, f"[{mode}]")
        print(ans[:900])
        print("Sources:", srcs)
