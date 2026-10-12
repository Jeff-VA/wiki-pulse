"""wiki-pulse: Wikipedia's live edits, on agentic-lakehouse.

- edits: the feed's events as article edits (a source transform)
- simulator: synthetic edits and alerts, for tests and work offline
- trends: the burst detector, a processor scoring each article in real time
- classifier: the news-or-not model, a pipeline step trained on Claude's labels
- explain: Claude's label and one-line explanation for a burst
"""
