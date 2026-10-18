"""Build the study's page: report.html with out/results.json filled in, written to out/classifier-study.html."""

from pathlib import Path

HERE = Path(__file__).parent

if __name__ == "__main__":
    page = (HERE / "report.html").read_text().replace("__RESULTS__", (HERE / "out" / "results.json").read_text())
    (HERE / "out" / "classifier-study.html").write_text(page)
    print(f"wrote {HERE / 'out' / 'classifier-study.html'}")
