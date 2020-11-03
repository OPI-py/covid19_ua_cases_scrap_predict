# from bs4 import BeautifulSoup
# import requests
import pandas as pd

# url = 'https://en.wikipedia.org/wiki/COVID-19_pandemic_in_Ukraine'

# source = requests.get(url)
# soup = BeautifulSoup(source.text, 'html.parser')

# tb1 = 'wikitable mw-datatable collapsible collapsed'
# table = soup.find_all('table', {'class': tb1})[1]

def make_df(table):
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

# tb2 = 'wikitable mw-datatable collapsible'
# table_july = soup.find('table', class_=tb2)
def make_july_df(table_july):
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
