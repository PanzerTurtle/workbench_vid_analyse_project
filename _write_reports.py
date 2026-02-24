import json
from json2html import json2html
from datetime import datetime


def generate_report(events: list, output):
    with open(output, "w") as report_file:
        report_file.write("=" * 60 + "\nEVENT ANALYSIS REPORT\n")
        report_file.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        report_file.write("=" * 60 + "\n\n")

        total = len(events)
        risks = {r: sum(e["risk_level"] == r for e in events) for r in ("high", "medium", "low")}

        report_file.write("SUMMARY\n")
        report_file.write(f"Total Frames: {total}\n")
        for k, v in risks.items():
            report_file.write(f"{k.title()} Risk: {v}\n")

        report_file.write("\nDETECTED EVENTS\n" + "-" * 60 + "\n")

        idx = 1
        for e in events:
            if e.get("event_detected"):
                report_file.write(
                    f"\nEvent #{idx}\n"
                    f"Time: {e['timestamp']}\n"
                    f"Risk: {e['risk_level'].upper()}\n"
                    f"Type: {e['event_type']}\n"
                    f"Details: {e['description']}\n"
                )
                idx += 1

        if idx == 1:
            report_file.write("\nNo events detected.\n")

    print(f"Report saved: {output}")
    

def json2html_convert_event_only(json_input: list, html_output: str):
    with open(json_input, "r") as f:
        events = json.load(f)

    filtered = [e for e in events if e.get("event_detected") is True]

    cleaned = []
    for e in filtered:
        e = dict(e)  # copy
        e.pop("event_detected", None)
        cleaned.append(e)

    with open(html_output, "w") as f:
        f.write("<html><body>\n")

        if cleaned:
            f.write(json2html.convert(cleaned))
        else:
            f.write("<p>No security-relevant events detected.</p>")

        f.write("\n</body></html>")
    
    print(f"Converted {json_input} to {html_output} sucessfully.")
    
  
def json2html_convert_all(json_input: list, html_output: str):
    with open(json_input, "r") as f:
        events = json.load(f)

    with open(html_output, "w") as f:
        f.write("<html><body>\n")
        f.write(json2html.convert(events))
        f.write("\n</body></html>")
    
    print(f"Converted {json_input} to {html_output} sucessfully.")
    
    
if __name__ == "__main__":
    json2html_convert_all("./reports/events_raw.json", "./reports/events_report_table.html")