

import pandas as pd


def make_df(table):
    """
    Scraping necessary data from 'mw-datatable collapsible collapsed'.
    Making dataframe with it.
    Required fields (Date, New cases, Total cases, New deaths, Total deaths).
    """
    rows = table.find_all('tr')
    columns = [v.text.replace('\n', '') for v in rows[2].find_all('th')]
    columns = columns[0:1] + columns[2:6]
    df = pd.DataFrame(columns = columns)

    for i in range(3, len(rows)):
        tds = rows[i].find_all('td')

        if len(tds) > 0:
            values = [tds[0].text.replace('\n', ''),
                tds[2].text.replace('\n', '').replace(',', ''),
                tds[3].text.replace('\n', '').replace(',', ''),
                tds[4].text.replace('\n', '').replace(',', ''),
                tds[5].text.replace('\n', '').replace(',', '')]
        else:
            values = [td.text for td in tds]

        df = df.append(pd.Series(values, index=columns), ignore_index=True)
    return df


def make_july_df(table_july):
    """
    Scraping necessary data from 'wikitable mw-datatable collapsible'.
    Making datafram with it.
    Required fields (Date, New cases, Total cases, New deaths, Total deaths)
    """
    rows = table_july.find_all('tr')
    columns = [v.text.replace('\n', '') for v in rows[2].find_all('th')]
    columns = columns[0:1] + columns[2:6]
    df = pd.DataFrame(columns = columns)

    for i in range(3, 25):
        tds = rows[i].find_all('td')

        if len(tds) > 0:
            values = [tds[0].text.replace('\n', ''),
                tds[2].text.replace('\n', '').replace(',', ''),
                tds[3].text.replace('\n', '').replace(',', ''),
                tds[4].text.replace('\n', '').replace(',', ''),
                tds[5].text.replace('\n', '').replace(',', '')]
        else:
            values = [td.text for td in tds]
        df = df.append(pd.Series(values, index=columns), ignore_index=True)
    return df
