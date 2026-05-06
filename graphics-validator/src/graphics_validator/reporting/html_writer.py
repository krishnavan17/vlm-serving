"""HTML summary writer for validation reports."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, BaseLoader

if TYPE_CHECKING:
    from graphics_validator.reporting.report import Report

_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Graphics Validator Report — {{ run_id }}</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 1rem; background: #0f0f0f; color: #e0e0e0; }
    h1 { color: #f0f0f0; }
    .summary { background: #1a1a1a; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; }
    .event { border: 1px solid #333; border-radius: 6px; margin-bottom: 1rem; padding: 1rem; background: #161616; }
    .event-header { display: flex; gap: 1rem; align-items: center; margin-bottom: 0.5rem; }
    .severity-critical { color: #ff4444; font-weight: bold; }
    .severity-high     { color: #ff8800; font-weight: bold; }
    .severity-medium   { color: #ffcc00; }
    .severity-low      { color: #aaaaaa; }
    .thumb { max-width: 320px; border-radius: 4px; border: 1px solid #444; }
    .flags { font-size: 0.85em; font-family: monospace; background: #111; padding: 0.5rem; border-radius: 4px; }
    .vlm-block { margin-top: 0.5rem; font-size: 0.85em; background: #0a1a0a; padding: 0.5rem; border-radius: 4px; }
    .no-events { color: #888; font-style: italic; }
  </style>
</head>
<body>
  <h1>🎮 Graphics Validator Report</h1>
  <div class="summary">
    <strong>Run ID:</strong> {{ run_id }}<br>
    <strong>Total flagged frames:</strong> {{ events|length }}<br>
  </div>

  {% if events %}
    {% for ev in events %}
    <div class="event">
      <div class="event-header">
        <strong>Frame {{ ev.frame_idx }}</strong>
        <span>@ {{ "%.2f"|format(ev.timestamp_s) }}s</span>
        {% set max_sev = ev.flags | map(attribute='severity') | list %}
        {% if 'critical' in max_sev %}
          <span class="severity-critical">CRITICAL</span>
        {% elif 'high' in max_sev %}
          <span class="severity-high">HIGH</span>
        {% elif 'medium' in max_sev %}
          <span class="severity-medium">MEDIUM</span>
        {% else %}
          <span class="severity-low">LOW</span>
        {% endif %}
      </div>

      {% if ev.thumb_path %}
        <img class="thumb" src="{{ ev.thumb_path }}" alt="frame {{ ev.frame_idx }}" loading="lazy">
      {% endif %}

      <div class="flags">
        {% for flag in ev.flags %}
          [{{ flag.layer }}] {{ flag.kind }} severity={{ flag.severity }} score={{ "%.3f"|format(flag.score) }}
          {% if flag.detail %} detail={{ flag.detail }}{% endif %}<br>
        {% endfor %}
      </div>

      {% if ev.vlm_response and not ev.vlm_response.get('skipped') %}
      <div class="vlm-block">
        <strong>VLM analysis:</strong><br>
        <pre>{{ ev.vlm_response | tojson(indent=2) }}</pre>
      </div>
      {% endif %}
    </div>
    {% endfor %}
  {% else %}
    <p class="no-events">No anomalies detected in this run.</p>
  {% endif %}
</body>
</html>
"""


def write_html(report: Report) -> Path:
    """Render the HTML summary for *report* and write it to ``index.html``.

    Returns
    -------
    Path
        Absolute path to the written ``index.html`` file.
    """
    env = Environment(loader=BaseLoader(), autoescape=True)
    env.filters["tojson"] = _tojson_filter
    template = env.from_string(_TEMPLATE)

    events_data = []
    for ev in report.events:
        d = {
            "frame_idx": ev.frame_idx,
            "timestamp_s": ev.timestamp_s,
            "flags": ev.flags,
            "thumb_path": ev.thumb_path,
            "vlm_response": ev.vlm_response,
        }
        events_data.append(d)

    html = template.render(run_id=report.run_dir.name, events=events_data)
    out_path = report.run_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _tojson_filter(value: object, indent: int = 2) -> str:
    import json

    return json.dumps(value, indent=indent, default=str)
