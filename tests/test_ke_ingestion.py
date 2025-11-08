import io
import pandas as pd

from etl.ingest_cbk import parse_cbk_auction_table
from etl.ingest_stanbic_pmi import write_csv as write_pmi_csv
from etl.ingest_knbs import parse_monthly_release_csv


def test_parse_cbk_html_sample():
    html = """
    <table>
      <tr><th>Date</th><th>Results</th></tr>
      <tr><td>12 Jan 2024</td><td>91-day T-Bill yield: 15.23% | 10Y Bond yield: 17.85%</td></tr>
      <tr><td>2024-02-09</td><td>91 day: 15.10% ; 10-year: 17.50%</td></tr>
    </table>
    """
    rows = parse_cbk_auction_table(html)
    assert len(rows) >= 2
    dates = [r[0].isoformat() for r in rows]
    assert '2024-01-12' in dates
    assert '2024-02-09' in dates


def test_parse_knbs_monthly_csv(tmp_path):
    p = tmp_path / 'knbs.csv'
    p.write_text("""year,month,cpi,unemployment
2024,1,130.2,5.6
2024,2,131.1,5.5
""")
    df = parse_monthly_release_csv(p)
    assert list(df.columns)[:1] == ['date']
    assert 'cpi' in df.columns and 'unemployment' in df.columns
    assert df.shape[0] == 2


def test_pmi_upload_pipeline(tmp_path, monkeypatch):
    # simulate write path
    from etl import ingest_stanbic_pmi as mod
    target = tmp_path / 'ke_pmi.csv'
    monkeypatch.setattr(mod, 'RAW_PATH', target)

    df = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01','2024-02-01']),
        'pmi': [49.5, 50.2]
    })
    write_pmi_csv(df)
    out = pd.read_csv(target)
    assert out.shape[0] == 2



