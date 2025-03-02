from xbbg import blp
import pandas as pd




def bmg_download_data(
        tickers: list,
        start_date: str,
        end_date: str,
        folder_path: str,
        base_currency: str = 'USD',
        field: str = 'last price'
        ) -> None:
    """ Download data from Bloomberg Terminal and save CSV files in the specified folder.
    VERY IMPORTANT!!!! RUN 
    "pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi" 
    ON TERMINAL, OTHERWISE THE FUNCTION WILL NOT WORK.

    Parameters:
    tickers (list): List of tickers to download.
    start_date (str): Start date in the format 'YYYY-MM-DD'.
    end_date (str): End date in the format 'YYYY-MM-DD'.
    folder_path (str): Folder path to save the CSV files.
    base_currency (str, optional): Base currency to convert the prices. Default is 'USD'.
    field (str, optional): Field to download from Bloomberg. Default is 'last price'.
    """

    for ticker in tickers:
        data = blp.bdh(
            tickers=ticker,
            start_date=start_date,
            end_date=end_date,
            Currency=base_currency,
            flds=field
        )

        data.rename(columns={data.columns[0]: "Date"}, inplace=True)
        data.rename(columns={data.columns[1]: f"Close {base_currency.upper()}"}, inplace=True)
        data.set_index("Date", inplace=True)
        data = data.iloc[1:]
        data.to_csv(f"{folder_path}/{ticker}.csv")


