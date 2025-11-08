from pathlib import Path
import pandas as pd


def main():
    raw = Path('data/raw')
    raw.mkdir(parents=True, exist_ok=True)

    cbk = pd.DataFrame({
        'date': ['2024-01-12','2024-02-09','2024-03-08','2024-04-05','2024-05-03','2024-06-07'],
        'ke_91d_bill_yield': [15.23,15.10,14.95,14.80,14.60,14.55],
        'ke_10y_bond_yield': [17.85,17.50,17.30,17.10,16.95,16.80],
    })
    cbk.to_csv(raw / 'ke_cbk.csv', index=False)

    knbs = pd.DataFrame({
        'date': ['2024-01-01','2024-02-01','2024-03-01','2024-04-01','2024-05-01','2024-06-01'],
        'cpi': [130.2,131.1,132.0,133.0,133.8,134.2],
        'unemployment': [5.6,5.5,5.6,5.5,5.4,5.5],
    })
    knbs.to_csv(raw / 'ke_knbs.csv', index=False)

    pmi = pd.DataFrame({
        'date': ['2024-01-01','2024-02-01','2024-03-01','2024-04-01','2024-05-01','2024-06-01'],
        'pmi': [49.5,50.2,50.8,49.9,49.7,50.1],
    })
    pmi.to_csv(raw / 'ke_pmi.csv', index=False)

    print('Seeded:', raw)


if __name__ == '__main__':
    main()



