import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from haversine import haversine

def result(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12):
    Extract_data = pd.read_csv('data.csv') # 데이터 베이스
    # 목적지 위도, 목적지 경도, 여행 기간, 경비, 인원, 나이, 거주지 위도, 거주지 경도, 활동 성향, 관광 성향, 일과 시작 시간, 일과 종료 시간
    # 총 12개의 데이터 셋
    User_data = pd.read_csv('user.csv') # 사용자 정보

    # 아래 데이터와 사용자 데이터가 일치하지 않는 데이터를 삭제한다.
    # 여행 기간, 인원 수

    Original_df = pd.DataFrame(Extract_data) # 본래 데이터 베이스의 데이터 프레임
    #User_df = pd.DataFrame(User_data) # 사용자 정보를 데이터 프레임 화
    User_df = pd.DataFrame({'destination_latitude' : a1,
                            'destination_longitude' : a2,
                            'schedule' : a3,
                            'expense' : a4,
                            'people' : a5,	
                            'age' : a6,
                        	'hometown_latitude' : a7,
                            'hometown_longitude' : a8,
                            'tendency_activity' : a9,
                            'tendency_preview' : a10,
                            'start_time' : a11,
                            'finsh_time' : a12,

    }, index = [0])
    U_data = User_df.loc[0] # User_df의 정보

    User_expense = User_df.loc[0]['expense']
    User_df.replace(to_replace=User_expense, value=User_expense/100000, inplace=True)
    id = []
    for i in range(len(Original_df)):
        id.append(i)

    Original_df['id'] = id
    #print(Original_df.loc[0]['expense'])

    o5 = User_df.loc[0]['schedule']

    for i in range(len(Original_df)):
        data = Original_df.loc[i]

        if (data['schedule'] == U_data['schedule']) & (data['people'] == U_data['people']):
            pass
        else:
            Original_df = Original_df.drop(index = i, axis = 0) # 여행 기간과 인원이 같은 행만을 남긴다.

    Original_df['expense'] = (Original_df['expense'])/100000

    #print(Original_df)

    o3 = User_df.loc[0]['people']
    Original_df = Original_df.drop('schedule', axis = 1)
    Original_df = Original_df.drop('people', axis = 1)

    User_df = User_df.drop('schedule', axis = 1)
    User_df = User_df.drop('people', axis = 1)

    #print(Original_df) # 일치하는 데이터만을 남겼으므로 필요없는 데이터의 열을 삭제한다.

    # 사용자 데이터와 유사한 데이터를 추출하기 위한 모델을 구현한다.
    # 모델 실행에는 여행 기간과 인원이 필요 없기에 제외한다.

    Sacrifice_df = Original_df # 모델을 만들기 위한 데이터 셋


    Sacrifice_df = Sacrifice_df.reset_index() # 행을 삭제한 후 index를 초기화한다.
    Sacrifice_df = Sacrifice_df.drop('index', axis = 1) # 필요없는 번호는 삭제한다.

    User_df['destination_distance'] = 0 # 사용자와 표본의 목적지 거리 값 추가
    Sacrifice_df['destination_distance'] = 0

    User_df['hometown_distance'] = 0 # 사용자와 표본의 거주지 거리 값 추가
    Sacrifice_df['hometown_distance'] = 0

    # 사용자의 목적지 위, 경도와 표본의 목적지 위, 경도를 비교하여 거리를 df에 추가
    User_D_latitude = User_df.iloc[0]['destination_latitude'] # 사용자의 목적지 위도
    User_D_longitude = User_df.iloc[0]['destination_longitude'] # 사용자의 목적지 경도
    User_D_position = (User_D_latitude, User_D_longitude) # 사용자의 목적지 좌표
    D_distance = [] # df에 추가할 목적지 거리 값 list

    # 사용자의 거주지 위, 경도와 표본의 목적지 위, 경도를 비교하여 거리를 df에 추가
    User_H_latitude = User_df.iloc[0]['hometown_latitude'] # 사용자의 거주지 위도
    User_H_longitude = User_df.iloc[0]['hometown_longitude'] # 사용자의 거주지 경도
    User_H_position = (User_H_latitude, User_H_longitude) # 사용자의 거주지 좌표
    H_distance = [] # df에 추가할 거주 거리 값 list



    for i in range(len(Sacrifice_df)):

        Sacrifice_D_latitude = Sacrifice_df.iloc[i]['destination_latitude'] # 표본의 목적지 위도
        Sacrifice_D_longitude = Sacrifice_df.iloc[i]['destination_longitude'] # 표본의 목적지 경도

        Sacrifice_D_position = (Sacrifice_D_latitude, Sacrifice_D_longitude) # 표본의 목적지 좌표

        D_distance.append(haversine(User_D_position, Sacrifice_D_position, unit="km"))

        Sacrifice_H_latitude = Sacrifice_df.iloc[i]['hometown_latitude']  # 표본의 거주지 위도
        Sacrifice_H_longitude = Sacrifice_df.iloc[i]['hometown_longitude']  # 표본의 거주지 경도

        Sacrifice_H_position = (Sacrifice_H_latitude, Sacrifice_H_longitude)  # 표본의 거주지 좌표

        H_distance.append(haversine(User_H_position, Sacrifice_H_position, unit="km"))

    Sacrifice_df['destination_distance'] = D_distance # 사용자와 표본 목적지 거리 값 추가 완료
    Sacrifice_df['hometown_distance'] = H_distance # 사용자와 표본 거주지 거리 값 추가 완료

    idx_nm = Sacrifice_df[Sacrifice_df['destination_distance'] > 30].index # 거주지 거리가 30km가 넘는 표본들은 제외한다.
    Sacrifice_df = Sacrifice_df.drop(idx_nm)

    Sacrifice_df = Sacrifice_df.reset_index() # 행을 삭제한 후 index를 초기화한다.

    Sacrifice_df = Sacrifice_df.drop('index', axis = 1)

    o1 = (User_df.loc[0]['destination_latitude'])
    o2 = User_df.loc[0]['destination_longitude']


    #print(Sacrifice_df)
    Sacrifice_df = Sacrifice_df.drop('destination_latitude', axis = 1)
    Sacrifice_df = Sacrifice_df.drop('destination_longitude', axis = 1)
    Sacrifice_df = Sacrifice_df.drop('hometown_latitude', axis = 1)
    Sacrifice_df = Sacrifice_df.drop('hometown_longitude', axis = 1)

    User_df = User_df.drop('destination_latitude', axis = 1)
    User_df = User_df.drop('destination_longitude', axis = 1)
    User_df = User_df.drop('hometown_latitude', axis = 1)
    User_df = User_df.drop('hometown_longitude', axis = 1)

    #print(Sacrifice_df.loc[0])
    #print(User_df.loc[0])

    Sacrifice2_df = Sacrifice_df
    Sacrifice2_df = Sacrifice2_df.drop('id', axis = 1)

    #Reference_value = np.array(User_df.loc[0])
    #Comparison_value = np.array(Sacrifice_df.loc[2])
    #cos_sim = cosine_similarity([Comparison_value], [Reference_value])
    #print(cos_sim)

    cos_sim_array = np.array([])
    Reference_value = np.array(User_df.loc[0]) # 사용자 데이터를 기준으로 비교한다.
    ooo = {}
    for i in range(len(Sacrifice_df)):
        Comparison_value = np.array(Sacrifice2_df.loc[i]) # 비교할 데이터 셋의 데이터
        #print(Comparison_value)
        cos_sim = cosine_similarity([Comparison_value], [Reference_value])
        a = cos_sim.tolist()
        ooo[a[0][0]]= Sacrifice_df.loc[i]['id']

    ooo = sorted(ooo.items(), reverse=True)

    import os
    import openai
    openai.api_key = "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"

    o4 = User_df.loc[0]['expense']

    a1=str(o1) #여행지 위도
    a2=str(o2) #여행지 경도
    b=str(o3) # 인원수
    c=str(o4*100000) #돈
    d=str(o5) #여행일정
    e="위도",a1,",경도",a2+"에 해당하는 지역에서 ",b,"명이서",c,"원으로",d,"일 여행계획 추천해줘"

    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{e}"}
    ]
    )
    li=[]
    if len(ooo)<int(float(b)):
        return 0
    else:
        for i in range(int(float(b))-1):
            li.append(str(ooo[i+1][1]))
        li.append(str(completion.choices[0].message['content'].strip()))
        return li    




'''print(completion.choices[0].message['content'].strip())
for i in range(int(float(b))-1):
    print('여행을 같이 갈 친구의 번호 : ')
   print(int(ooo[i+1][1]))

def result():
    li=[]
    for i in range(int(float(b))-1):
        li.append(str(ooo[i+1][1]))
    li.append(str(completion.choices[0].message['content'].strip()))
    return li
'''
