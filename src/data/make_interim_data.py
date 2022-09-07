import pandas as pd


def preparing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.strip() for col in df.columns]  # type: ignore
    return df


if __name__ == "__main__":
    df = pd.read_excel('../../data/raw/data.xlsx')
    prepared_orders = preparing_pipeline(df)
    prepared_orders.to_excel('../../data/interim/data.xlsx', index=False)
