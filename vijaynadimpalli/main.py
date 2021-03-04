import pandas as pd
import numpy as np
from partner_selection import PartnerSelection
import time
import mlfinlab
from ps_utils import get_sector_data

df = pd.read_csv('./data/data.csv', parse_dates=True, index_col='Date').dropna().iloc[:, :] #Reducing universe to 60 stocks
df = df['2016'] #Taking 12 month data as metioned in the paper
ps = PartnerSelection(df)

constituents = pd.read_csv('./data/constituents-detailed.csv', index_col='Symbol')

#ps.extremal()  #18 seconds per target
# ['A', 'AMP', 'BLK', 'IVZ']
# ['AAL', 'UAL', 'DAL', 'LUV']
# ['AAP', 'AZO', 'LOW', 'HD']
# ['AAPL', 'GOOGL', 'GOOG', 'MSFT']
# ['ABBV', 'AMGN', 'CAH', 'MCK']

#ps.extremal_multiprocess()

#ps.plot_all_target_measures(target='A', procedure='extended')


#ps.plot_correlation()

#print(ps.traditional_vectorized())
#[['A', 'TMO', 'PKI', 'WAT'], ['AAL', 'UAL', 'DAL', 'LUV'], ['AAP', 'TT', 'PH', 'ETN'], ['AAPL', 'TXN', 'ADI', 'INTC'], ['ABBV', 'AMGN', 'REGN', 'BIIB']]


#print(ps.traditional(1)) #takes 40 seconds per target
#[['A', 'TMO', 'PKI', 'WAT'], ['AAL', 'UAL', 'DAL', 'LUV'], ['AAP', 'TT', 'PH', 'ETN'], ['AAPL', 'TXN', 'ADI', 'INTC'], ['ABBV', 'AMGN', 'REGN', 'BIIB']]



#print(ps.extended_vectorized())
#[['A', 'TMO', 'PKI', 'WAT'], ['AAL', 'UAL', 'DAL', 'LUV'], ['AAP', 'AZO', 'ORLY', 'ULTA'], ['AAPL', 'GOOGL', 'GOOG', 'MSFT'], ['ABBV', 'AMGN', 'REGN', 'BIIB']]
# est1 [0.70071437 0.66445788 0.6348848  ... 0.56125229 0.44962331 0.45367421]
# est2 [0.7480098  0.6814423  0.65838103 ... 0.5683275  0.50264483 0.48187037]
# est3 [-2.99927335 -2.9992825  -2.99928789 ... -2.999305   -2.99932067
#  -2.99932272]
# ['A', 'TMO', 'PKI', 'WAT']
# est1 [0.73894415 0.73279143 0.67236117 ... 0.61573199 0.58640538 0.58710008]
# est2 [0.77417177 0.71804952 0.62625584 ... 0.63918809 0.61260781 0.63121752]
# est3 [-2.99926681 -2.99927294 -2.99928718 ... -2.99929103 -2.9992965
#  -2.99929491]
# ['AAL', 'UAL', 'DAL', 'LUV']
# est1 [0.48394775 0.54193125 0.5376858  ... 0.39117871 0.38590975 0.41218557]
# est2 [0.53263034 0.5602267  0.54150317 ... 0.45435388 0.44333247 0.46693958]
# est3 [-2.99931408 -2.99930558 -2.99930793 ... -2.99933072 -2.99933239
#  -2.99932738]
# ['AAP', 'AZO', 'ORLY', 'ULTA']
# est1 [0.61038412 0.60988793 0.60082431 ... 0.40875047 0.36665277 0.4197442 ]
# est2 [0.59690581 0.61348135 0.58191815 ... 0.42175352 0.38728735 0.44959034]
# est3 [-2.99929671 -2.99929503 -2.99929805 ... -2.99933232 -2.9993394
#  -2.99932882]
# ['AAPL', 'GOOGL', 'GOOG', 'MSFT']
# est1 [0.58208769 0.5672097  0.57365798 ... 0.44649239 0.35562415 0.39057755]
# est2 [0.59063258 0.5761572  0.59973694 ... 0.42505645 0.36839786 0.38835952]
# est3 [-2.9992982  -2.9993011  -2.99929858 ... -2.99932903 -2.9993433
#  -2.99933747]
# ['ABBV', 'AMGN', 'REGN', 'BIIB']
# [['A', 'TMO', 'PKI', 'WAT'], ['AAL', 'UAL', 'DAL', 'LUV'], ['AAP', 'AZO', 'ORLY', 'ULTA'], ['AAPL', 'GOOGL', 'GOOG', 'MSFT'], ['ABBV', 'AMGN', 'REGN', 'BIIB']]

#print(ps.extended(1))

#ps.geometric_vectorized()
# [53.84724318 59.21223928 62.8272696  ... 69.47369565 78.14625353
#  78.4739646 ]
# ['A', 'TMO', 'PKI', 'WAT']

#ps.geometric(5)


#print(ps.traditional_multiprocess()) #takes 12 seconds per target

#print(ps.extended_multiprocess()) #takes 40 seconds per target

#print(ps.geometric_multiprocess()) #takes 40 seconds per target

# ps.plot_selected_pairs([['A', 'TMO', 'PKI', 'WAT'], ['AAL', 'UAL', 'DAL', 'LUV'],
#                         ['AAP', 'TT', 'PH', 'ETN'], ['AAPL', 'GOOGL', 'GOOG', 'MSFT'], ['ABBV', 'WAT', 'A', 'TMO']])
#
#
# ps.plot_selected_pairs([['A', 'TMO', 'PKI', 'WAT'], ['AAL', 'UAL', 'DAL', 'LUV'],
#                         ['AAP', 'TT', 'PH', 'ETN'], ['AAPL', 'TXN', 'ADI', 'INTC'], ['ABBV', 'AMGN', 'REGN', 'BIIB']])



# ps.plot_selected_pairs([['A' 'TMO' 'PKI' 'WAT'],
#  ['AAL' 'UAL' 'DAL' 'LUV'],
#  ['AAP' 'TT' 'PH' 'ETN'],
#  ['AAPL' 'TXN' 'ADI' 'INTC'],
#  ['ABBV' 'AMGN' 'REGN' 'BIIB'],
#  ['ABC' 'JPM' 'TFC' 'PNC'],
#  ['ABMD' 'WAT' 'PKI' 'TMO'],
#  ['ABT' 'TMO' 'A' 'WAT'],
#  ['ACN' 'ADP' 'JKHY' 'FISV'],
#  ['ADBE' 'GOOG' 'MSFT' 'GOOGL'],
#  ['ADI' 'TXN' 'MCHP' 'INTC'],
#  ['ADM' 'BEN' 'TROW' 'IVZ'],
#  ['ADP' 'PAYX' 'FISV' 'JKHY'],
#  ['ADSK' 'C' 'BK' 'SCHW'],
#  ['AEE' 'XEL' 'WEC' 'CMS'],
#  ['AEP' 'XEL' 'DUK' 'WEC'],
#  ['AES' 'AEP' 'CMS' 'XEL'],
#  ['AFL' 'UNM' 'PRU' 'LNC'],
#  ['AIG' 'PRU' 'MET' 'LNC'],
#  ['AIV' 'UDR' 'AVB' 'ESS']])

