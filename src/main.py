import requests
import pickle
import requests
import json
import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from skimage import io
from sklearn.preprocessing import MinMaxScaler

api_key = 'api_key=RGAPI-dd5dc044-5799-44d0-81d7-4640727c738a'


def update_champ():
    r = requests.get('https://ddragon.leagueoflegends.com/api/versions.json')
    current_version = r.json()[0]

    r = requests.get('http://ddragon.leagueoflegends.com/cdn/'
                     + '{}/data/ko_KR/champion.json'.format(current_version))

    parsed_data = r.json()
    info_df = pd.DataFrame(parsed_data)

    champ_dic = {}
    for i, champ in enumerate(info_df.data):
        champ_dic[i] = pd.Series(champ)

    champ_df = pd.DataFrame(champ_dic).T
    champ_df = champ_df[['id', 'key', 'name']]
    print(champ_df)
    r = champ_df.to_csv('./db/champion_list.csv')

    return


def update_summoner():
    sname = pd.DataFrame(columns=['summoner_name','score'])
    chall = 'https://kr.api.riotgames.com/lol/league/v4/challengerleagues' \
        + '/by-queue/RANKED_SOLO_5x5' + '?' + api_key

    r = requests.get(chall)
    cnt = 0

    for i in r.json()['entries']:
        sname.loc[cnt] = i['summonerName'],  i['leaguePoints']
        cnt = cnt + 1
        
    sname = sname.sort_values(by='score', ascending =False)
    sname.to_csv('./db/summonerName_list.csv')


def update_puuId():
    sname = pd.read_csv('./db/summonerName_list.csv', index_col=0)
    accountid = pd.DataFrame(columns=['puuid'])
    cnt = 0
    for i in range(len(sname)):
        print(sname['summoner_name'].loc[i], 'loading data')
        url_id = "https://kr.api.riotgames.com/lol/summoner/v4/" \
            + 'summoners/by-name/' + sname['summoner_name'].loc[i] + '?' + api_key
        r = requests.get(url_id)

        if r.status_code == 404:
            continue
        while r.status_code == 429: #rate limit exceeded
            time.sleep(3)
            r = requests.get(url_id)

        accountid.loc[cnt] = r.json()['puuid']
        cnt = cnt + 1

    print('puuID update done!')
    accountid.to_csv('./db/puuId_list.csv')


def update_game():
    puuid = pd.read_csv('./db/puuId_list.csv', index_col=0)
    game = pd.DataFrame(columns=['gameId'])
    cnt = 0

    for i in range(len(puuid)):
        print(i, ' of ', len(puuid))
        url_gameid = f"https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid['puuid'].iloc[i]}/ids?type=ranked&start=0&count=50&{api_key}" # only rank game, 50 data at a time 
        r = requests.get(url_gameid)

        if r.status_code == 404: # data not found
            continue
        while r.status_code == 429:  #rate limit exceeded
            time.sleep(3)
            r = requests.get(url_gameid)

        for j in r.json(): 
            game.loc[cnt] = j
            cnt = cnt + 1

    game.drop_duplicates()
    # game.reset_index(inplace=True)
    game.to_csv('./db/gameId_list.csv',mode='a', header=['gameId'])




def make_data():
    games = pd.read_csv('./db/gameId_list.csv', index_col=0)
    
    num_game = len(games)

    for i in range(11421, num_game):
        game_id = games['gameId'].iloc[i]
        #lane = pd.DataFrame(columns=['bTOP', 'bJUG', 'bMID', 'bADC', 'bSUP','rTOP', 'rJUG', 'rMID', 'rADC', 'rSUP', 'win'])
        api_url = f'https://asia.api.riotgames.com/lol/match/v5/matches/{game_id}?{api_key}'
        r = requests.get(api_url)

        while r.status_code == 429:
            time.sleep(3)
            r = requests.get(api_url)

        gamers = pd.DataFrame((r.json()['info']['participants']))
        pos_list = [0 for i in range(11)]
        
        print(f'Making data from {game_id}...{i} of {num_game}') #Progress tracking
        cnt = 0

        for idx in gamers.index:
            if cnt <= 4:
                if gamers.loc[idx,'teamPosition'] == 'TOP' :
                    pos_list[0] = gamers.loc[idx,'championName']
                if gamers.loc[idx,'teamPosition'] == 'JUNGLE' :
                    pos_list[1] = gamers.loc[idx,'championName']
                if gamers.loc[idx,'teamPosition'] == 'MIDDLE' :
                    pos_list[2] = gamers.loc[idx,'championName']
                if gamers.loc[idx,'teamPosition'] == 'BOTTOM' :
                    pos_list[3] = gamers.loc[idx,'championName']
                if gamers.loc[idx,'teamPosition'] == 'UTILITY' :
                    pos_list[4] = gamers.loc[idx,'championName']

            else:
                if gamers.loc[idx,'teamPosition'] == 'TOP' :
                    pos_list[5] = gamers.loc[idx,'championName']
                if gamers.loc[idx,'teamPosition'] == 'JUNGLE' :
                    pos_list[6] = gamers.loc[idx,'championName']
                if gamers.loc[idx,'teamPosition'] == 'MIDDLE' :
                    pos_list[7] = gamers.loc[idx,'championName']
                if gamers.loc[idx,'teamPosition'] == 'BOTTOM' :
                    pos_list[8] = gamers.loc[idx,'championName']
                if gamers.loc[idx,'teamPosition'] == 'UTILITY' :
                    pos_list[9] = gamers.loc[idx,'championName']
            cnt += 1

        pos_list[10] = gamers.loc[0,'win']
        
        lane = pd.DataFrame([pos_list], columns=['bTOP', 'bJUG', 'bMID', 'bADC', 'bSUP','rTOP', 'rJUG', 'rMID', 'rADC', 'rSUP', 'win'])
        lane.to_csv('./db/gameData.csv',mode='a', header=False)
    return



#  MAIN  ##################################################################

torch.manual_seed(1)

x_data = []

for i in range(6):
    blue = []
    red = []
    
    rand = random.sample(range(15),5)
    for j in rand:
        temp = [0 for k in range(15)]
        temp[j] = 1
        blue.append(temp)

    rand = random.sample(range(15),5)
    for j in rand:
        temp = [0 for k in range(15)]
        temp[j] = 1
        red.append(temp)
    
    blue.extend(red)
    x_data.append(blue)

y_data = [0,0,0,1,1,1]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)


"""
TextRNN Parameter
"""
batch_size = 6
n_step = 2  # 학습 하려고 하는 문장의 길이 - 1
n_hidden = 5  # 은닉층 사이즈
n_class = 15
dtype = torch.float

input_batch = x_train
target_batch = y_train

"""
TextRNN
"""
class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()

        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.3)
        self.W = nn.Parameter(torch.randn([n_hidden, n_class]).type(dtype))
        self.b = nn.Parameter(torch.randn([n_class]).type(dtype))
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, hidden, X):
        X = X.transpose(0, 1)
        outputs, hidden = self.rnn(X, hidden)
        outputs = outputs[-1]  # 최종 예측 Hidden Layer
        model = torch.mm(outputs, self.W) + self.b  # 최종 예측 최종 출력 층
        return model
	

"""
Training
"""
model = TextRNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    hidden = torch.zeros(1, batch_size, n_hidden, requires_grad=True)
    output = model(hidden, input_batch)
    loss = criterion(output, target_batch)

    if (epoch + 1) % 100 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

input = [sen.split()[:2] for sen in sentences]

hidden = torch.zeros(1, batch_size, n_hidden, requires_grad=True)
predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]









'''champs = pd.read_csv('./db/champion_list.csv', index_col=0)

data = pd.read_csv('./db/gameData.csv', index_col=0)
temp = pd.get_dummies(champs['id'])

torch.manual_seed(1)

x_data = []

for i in range(6):
    blue = []
    red = []
    
    rand = random.sample(range(15),5)
    for j in rand:
        temp = [0 for k in range(15)]
        temp[j] = 1
        blue.append(temp)

    rand = random.sample(range(15),5)
    for j in rand:
        temp = [0 for k in range(15)]
        temp[j] = 1
        red.append(temp)
    
    blue.extend(red)
    x_data.append(blue)

y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)


class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=15, out_features=10, bias=True),
            nn.ReLU(), 
            )

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=10, out_features=1, bias=True),
            nn.Sigmoid(), 
            )
    
    def forward(self, x):
        x = self.layer1(x) 
        x = self.layer2(x)
        return torch.sigmoid(x)

model = BinaryClassifier()
optimizer = optim.SGD(model.parameters(), lr=1)
loss = nn.BCELoss()

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = model(x_train)

    # cost 계산
    cost = loss(hypothesis, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5]) # 예측값이 0.5를 넘으면 True로 간주
        correct_prediction = prediction.float() == y_train # 실제값과 일치하는 경우만 True로 간주
        accuracy = correct_prediction.sum().item() / len(correct_prediction) # 정확도를 계산
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format( # 각 에포크마다 정확도를 출력
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))'''

###########################################################################
