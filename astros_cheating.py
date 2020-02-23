# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:06:56 2020

@author: Ken
"""

import pandas as pd 
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.discrete.discrete_model import Logit
import numpy as np
import seaborn as sns


df = pd.read_csv('astros_data.csv')

#create runs
df['astros_runs'] = df.Score.apply(lambda x: int(x[-1]) if x != '—' else '—' )
#Wins == 1 losses == 0 
df.Score = df.Score.apply(lambda x: 1 if 'Astros' in x else 0)
#drop playoff games
df = df.drop(df.tail(2).index)
###############################################################################
# Get hits and box score data 
###############################################################################

from bs4 import BeautifulSoup
import requests

result = requests.get('https://www.baseball-reference.com/teams/HOU/2017-schedule-scores.shtml#all_win_loss')
c = result.content

soup = BeautifulSoup(c)

links = soup.findAll('a', text = 'boxscore')

for i in links:
    print(i.get('href'))

website='https://www.baseball-reference.com'

###############################################################
"""
req_1 = requests.get(website+links[0].get('href'))
gm = req_1.content
gmbs4 = BeautifulSoup(gm)

date = gmbs4.find('div', class_='scorebox_meta')
dlist = date.find_all('div')
dlist[0].text
box_score = []
box_score.append(dlist[0].text)

table = gmbs4.find('table', class_='linescore nohover stats_table no_freeze')

for i in table.find_all('td'):
    if (i.text.strip not in ['via Sports Logos.net', 'About logos']) and (i.text.find("About logos") <1):
        box_score.append(i.text)
 """
###############################################################
        
def get_scores(links, website):
    data_out = []
    for i in links:
        row_data = []
        req = requests.get(website+i.get('href'))
        gm = req.content
        soup = BeautifulSoup(gm)
        
        date = soup.find('div', class_='scorebox_meta')
        dlist = date.find_all('div')
        dlist[0].text
        row_data.append(dlist[0].text)
        
        table = soup.find('table', class_='linescore nohover stats_table no_freeze')
        for i in table.find_all('td'):
            if (i.text.strip not in ['via Sports Logos.net', 'About logos']) and (i.text.find("About logos") <1) and (~i.text.startswith('Winning')):
                row_data.append(i.text)
        data_out.append(row_data)
    return data_out

box_scores = get_scores(links,website)                
box_scores[1]

home_game_bs = []

for i in box_scores:
    if i[1] != 'Houston Astros':
        home_game_bs.append(i)


df['scraped'] = home_game_bs
df = df[df.Bangs != '—']
df.Bangs = df.Bangs.astype(int)

def get_away_bs(plist):
    away_bs = []
    for i in plist[2:]:
        if len(i) <=2:
            away_bs.append(i)
        else: 
            return away_bs[:-3]

def get_away_bs_hre(plist):
    away_bs = []
    for i in plist[2:]:
        if len(i) <=2:
            away_bs.append(i)
        else: 
            return away_bs[-3:]

def get_home_bs(plist):
    away_bs = []
    if plist[-1].startswith('Winning'):
        plist = plist[:-1]
    for i in plist[:-1][::-1]:
        if len(i) <=2:
            away_bs.append(i)
        else: 
            return away_bs[::-1][:-3]

def get_home_bs_hre(plist):
    away_bs = []
    if plist[-1].startswith('Winning'):
        plist = plist[:-1]
    for i in plist[:-1][::-1]:
        if len(i) <=2:
            away_bs.append(i)
        else: 
            return away_bs[::-1][-3:]

        
df['away_bs'] = df.scraped.apply(lambda x: get_away_bs(x))
df['away_bs_hre'] = df.scraped.apply(lambda x: get_away_bs_hre(x))
df['home_bs'] = df.scraped.apply(lambda x: get_home_bs(x))
df['home_bs_hre'] = df.scraped.apply(lambda x: get_home_bs_hre(x))
df['astros_hits'] = df.home_bs_hre.apply(lambda x: x[1]).astype(int)
df['astros_runs'] =df.home_bs_hre.apply(lambda x: x[0]).astype(int)

def losing_in(xlst1,xlst2,inning):
    if np.sum(xlst1[:inning] > xlst2[:inning]):
        return 1
    else: return 0
    
def losing_t_in(xlst1,xlst2,inning):
    if np.sum(xlst1[:inning] >= xlst2[:inning]):
        return 1
    else: return 0
    
def t_in(xlst1,xlst2,inning):
    if np.sum(xlst1[:inning] == xlst2[:inning]):
        return 1
    else: return 0

def w_in(xlst1,xlst2,inning):
    if np.sum(xlst1[:inning] < xlst2[:inning]):
        return 1
    else: return 0
    
df['losing_1'] = df.apply(lambda x: losing_in(x.away_bs,x.home_bs,1), axis=1).astype(int)
df['losing_2'] = df.apply(lambda x: losing_in(x.away_bs,x.home_bs,2), axis=1).astype(int)
df['losing_3'] = df.apply(lambda x: losing_in(x.away_bs,x.home_bs,3), axis=1).astype(int)  
df['losing_4'] = df.apply(lambda x: losing_in(x.away_bs,x.home_bs,4), axis=1).astype(int)   
df['losing_5'] = df.apply(lambda x: losing_in(x.away_bs,x.home_bs,5), axis=1).astype(int)  
df['losing_6'] = df.apply(lambda x: losing_in(x.away_bs,x.home_bs,6), axis=1).astype(int)   
df['losing_7'] = df.apply(lambda x: losing_in(x.away_bs,x.home_bs,7), axis=1).astype(int)  
df['losing_8'] = df.apply(lambda x: losing_in(x.away_bs,x.home_bs,8), axis=1).astype(int)  
df['losing_9'] = df.apply(lambda x: losing_in(x.away_bs,x.home_bs,9), axis=1).astype(int)   

df['losing_t1'] = df.apply(lambda x: losing_t_in(x.away_bs,x.home_bs,1), axis=1).astype(int)
df['losing_t2'] = df.apply(lambda x: losing_t_in(x.away_bs,x.home_bs,2), axis=1).astype(int)
df['losing_t3'] = df.apply(lambda x: losing_t_in(x.away_bs,x.home_bs,3), axis=1).astype(int)  
df['losing_t4'] = df.apply(lambda x: losing_t_in(x.away_bs,x.home_bs,4), axis=1).astype(int)   
df['losing_t5'] = df.apply(lambda x: losing_t_in(x.away_bs,x.home_bs,5), axis=1).astype(int)  
df['losing_t6'] = df.apply(lambda x: losing_t_in(x.away_bs,x.home_bs,6), axis=1).astype(int)   
df['losing_t7'] = df.apply(lambda x: losing_t_in(x.away_bs,x.home_bs,7), axis=1).astype(int)  
df['losing_t8'] = df.apply(lambda x: losing_t_in(x.away_bs,x.home_bs,8), axis=1).astype(int)  
df['losing_t9'] = df.apply(lambda x: losing_t_in(x.away_bs,x.home_bs,9), axis=1).astype(int)   

df['tie1'] = df.apply(lambda x: t_in(x.away_bs,x.home_bs,1), axis=1).astype(int)
df['tie2'] = df.apply(lambda x: t_in(x.away_bs,x.home_bs,2), axis=1).astype(int)
df['tie3'] = df.apply(lambda x: t_in(x.away_bs,x.home_bs,3), axis=1).astype(int)  
df['tie4'] = df.apply(lambda x: t_in(x.away_bs,x.home_bs,4), axis=1).astype(int)   
df['tie5'] = df.apply(lambda x: t_in(x.away_bs,x.home_bs,5), axis=1).astype(int)  
df['tie6'] = df.apply(lambda x: t_in(x.away_bs,x.home_bs,6), axis=1).astype(int)   
df['tie7'] = df.apply(lambda x: t_in(x.away_bs,x.home_bs,7), axis=1).astype(int)  
df['tie8'] = df.apply(lambda x: t_in(x.away_bs,x.home_bs,8), axis=1).astype(int)  
df['tie9'] = df.apply(lambda x: t_in(x.away_bs,x.home_bs,9), axis=1).astype(int)  

df['win1'] = df.apply(lambda x: w_in(x.away_bs,x.home_bs,1), axis=1).astype(int)
df['win2'] = df.apply(lambda x: w_in(x.away_bs,x.home_bs,2), axis=1).astype(int)
df['win3'] = df.apply(lambda x: w_in(x.away_bs,x.home_bs,3), axis=1).astype(int)  
df['win4'] = df.apply(lambda x: w_in(x.away_bs,x.home_bs,4), axis=1).astype(int)   
df['win5'] = df.apply(lambda x: w_in(x.away_bs,x.home_bs,5), axis=1).astype(int)  
df['win6'] = df.apply(lambda x: w_in(x.away_bs,x.home_bs,6), axis=1).astype(int)   
df['win7'] = df.apply(lambda x: w_in(x.away_bs,x.home_bs,7), axis=1).astype(int)  
df['win8'] = df.apply(lambda x: w_in(x.away_bs,x.home_bs,8), axis=1).astype(int)  
df['win9'] = df.apply(lambda x: w_in(x.away_bs,x.home_bs,9), axis=1).astype(int)  
        
###############################################################################
# Do Analysis
###############################################################################

#correlation between bangs and runs 
df.astros_runs.corr(df.Bangs)
df.astros_hits.corr(df.Bangs)

sns.scatterplot(df.Bangs,df.astros_runs)
plt.title('Correlation Between Bangs and Runs')
plt.ylabel('Runs')
plt.xlabel('Bangs')

z = np.polyfit(df.Bangs, df.astros_runs, 1)
p = np.poly1d(z)
plt.plot(df.Bangs,p(df.Bangs),"r--")

plt.show()


df.astros_runs.corr(df.Bangs)
df.astros_hits.corr(df.Bangs)

sns.scatterplot(df.Bangs,df.astros_hits)
plt.title('Correlation Between Bangs and Hits')
plt.ylabel('Hits')
plt.xlabel('Bangs')

z = np.polyfit(df.Bangs, df.astros_hits, 1)
p = np.poly1d(z)
plt.plot(df.Bangs,p(df.Bangs),"r--")

plt.show()

#bangs by win / loss
pd.pivot_table(df, index = 'Score', values='Bangs')

#create intercept, needed for OLS regression 
df['intercept'] = 1
# xvar 
X = df[['Bangs','intercept']]

#scores 
y = df.astros_runs.values

# linear regression 
model_ols = sm.OLS(df.astros_runs,X).fit()
model_ols.summary()

model_ols = sm.OLS(df.astros_hits,X).fit()
model_ols.summary()

#logistic regression for wins 
model_logit = Logit(df.Score,X).fit()
model_logit.summary()

#although anecdotal, a statistic similar to saves may add fuel to this fire

win_bang = pd.pivot_table(df,index='Score', values = 'Bangs')

bangs_1 = pd.pivot_table(df, index='losing_1',values='Bangs', aggfunc = ['mean','count'])
bangs_2 = pd.pivot_table(df, index='losing_2',values='Bangs', aggfunc = ['mean','count'])
bangs_3 = pd.pivot_table(df, index='losing_3',values='Bangs', aggfunc = ['mean','count'])
bangs_4 = pd.pivot_table(df, index='losing_4',values='Bangs', aggfunc = ['mean','count'])
bangs_5 = pd.pivot_table(df, index='losing_5',values='Bangs', aggfunc = ['mean','count'])
bangs_6 = pd.pivot_table(df, index='losing_6',values='Bangs', aggfunc = ['mean','count'])
bangs_7 = pd.pivot_table(df, index='losing_7',values='Bangs', aggfunc = ['mean','count'])
bangs_8 = pd.pivot_table(df, index='losing_8',values='Bangs', aggfunc = ['mean','count'])
bangs_9 = pd.pivot_table(df, index='losing_9',values='Bangs', aggfunc = ['mean','count'])

plt.plot(columns,bangs_losing)

bangs_1t = pd.pivot_table(df, index='losing_t1',values='Bangs', aggfunc = ['mean','count'])
bangs_2t = pd.pivot_table(df, index='losing_t2',values='Bangs', aggfunc = ['mean','count'])
bangs_3t = pd.pivot_table(df, index='losing_t3',values='Bangs', aggfunc = ['mean','count'])
bangs_4t = pd.pivot_table(df, index='losing_t4',values='Bangs', aggfunc = ['mean','count'])
bangs_5t = pd.pivot_table(df, index='losing_t5',values='Bangs', aggfunc = ['mean','count'])
bangs_6t = pd.pivot_table(df, index='losing_t6',values='Bangs', aggfunc = ['mean','count'])
bangs_7t = pd.pivot_table(df, index='losing_t7',values='Bangs', aggfunc = ['mean','count'])
bangs_8t = pd.pivot_table(df, index='losing_t8',values='Bangs', aggfunc = ['mean','count'])
bangs_9t = pd.pivot_table(df, index='losing_t9',values='Bangs', aggfunc = ['mean','count'])

bangs_1tie = pd.pivot_table(df, index='tie1',values='Bangs', aggfunc = ['mean','count'])
bangs_2tie = pd.pivot_table(df, index='tie2',values='Bangs', aggfunc = ['mean','count'])
bangs_3tie = pd.pivot_table(df, index='tie3',values='Bangs', aggfunc = ['mean','count'])
bangs_5tie = pd.pivot_table(df, index='tie5',values='Bangs', aggfunc = ['mean','count'])
bangs_7tie = pd.pivot_table(df, index='tie7',values='Bangs', aggfunc = ['mean','count'])

bangs_1win = pd.pivot_table(df, index='win1',values='Bangs', aggfunc = ['mean','count'])
bangs_2win = pd.pivot_table(df, index='win2',values='Bangs', aggfunc = ['mean','count'])
bangs_3win = pd.pivot_table(df, index='win3',values='Bangs', aggfunc = ['mean','count'])
bangs_5win = pd.pivot_table(df, index='win5',values='Bangs', aggfunc = ['mean','count'])
bangs_7win = pd.pivot_table(df, index='win7',values='Bangs', aggfunc = ['mean','count'])

gms_won =pd.pivot_table(df, index='Score', values = ['losing_1', 'losing_2', 'losing_3','losing_4','losing_5','losing_6','losing_7','losing_8','losing_9'], aggfunc = 'sum').iloc[1,:]
gms_tot = pd.pivot_table(df, index='Score', values = ['losing_1', 'losing_2', 'losing_3','losing_4','losing_5','losing_6','losing_7','losing_8','losing_9'], aggfunc = 'sum').sum(axis=0)
pd.pivot_table(df, index='Score', values = ['losing_t1', 'losing_t2', 'losing_t3','losing_t4','losing_t5','losing_t6','losing_t7','losing_t8','losing_t9'], aggfunc = 'sum')
pd.pivot_table(df, index='Score', values = ['tie1', 'tie2', 'tie3','tie4','tie5','tie6','tie7','tie8','tie9'], aggfunc = 'sum')

winwin = pd.pivot_table(df, index='Score', values = ['win1', 'win2', 'win3','win4','win5','win6','win7','win8','win9'], aggfunc = 'sum').iloc[1,:]
winwin_tot =pd.pivot_table(df, index='Score', values = ['win1', 'win2', 'win3','win4','win5','win6','win7','win8','win9'], aggfunc = 'sum').sum(axis=0)

pd.pivot_table(df, index='Score', values = 'Bangs', columns =['losing_1','losing_2'])

# Graphs, total bangs by situation in inning 
bangs_losing = [bangs_1.iloc[1,0],bangs_2.iloc[1,0],bangs_3.iloc[1,0],bangs_4.iloc[1,0],bangs_5.iloc[1,0],bangs_6.iloc[1,0],bangs_7.iloc[1,0],bangs_8.iloc[1,0],bangs_9.iloc[1,0]]
bangs_winning = [bangs_1t.iloc[0,0],bangs_2t.iloc[0,0],bangs_3t.iloc[0,0],bangs_4t.iloc[0,0],bangs_5t.iloc[0,0],bangs_6t.iloc[0,0],bangs_7t.iloc[0,0],bangs_8t.iloc[0,0],bangs_9t.iloc[0,0]]
columns = ['1st','2nd','3rd','4th','5th','6th','7th','8th','9th']
plt.plot(columns,bangs_losing, label = 'losing in inning')
plt.plot(columns,bangs_winning, label = 'winning in inning')
plt.xlabel('Inning')
plt.ylabel('Total Bangs in Game')
plt.legend()

# Win % by score in inning relative to expected (losing & winning)
expected_win = [.655,.675,.704,.735,.773,.845,.877]
expected_win_actual = np.array([1,1,1,1,1,1,1]) - np.array(expected_win)
actual_win = gms_won/gms_tot
columns_7 = ['1st','2nd','3rd','4th','5th','6th','7th']

plt.plot(columns_7,expected_win_actual, label = 'expected win % when behind in inning')
plt.plot(columns_7,actual_win[:7], label = 'actual win % when behind in inning')
plt.xlabel('Inning')
plt.ylabel('Win Pct')
plt.legend()

exp_win_wins = [.712,.725,.750,.775,.804,.845,.890]
actual_win_win = winwin / winwin_tot

plt.plot(columns_7,exp_win_wins, label = 'expected win % when ahead in inning')
plt.plot(columns_7,actual_win_win[:7], label = 'actual win % when ahead in inning')
plt.xlabel('Inning')
plt.ylabel('Win Pct')
plt.legend()

