from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
from functools import lru_cache
import copy

app = FastAPI()

# CORS設定（フロントエンドからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切なドメインに制限してください
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. 定数・設定データ定義
# ==========================================

# 各サブステータスの出現値
cr_list = [6.3, 6.9, 7.5, 8.1, 8.7, 9.3, 9.9, 10.5]
cd_list = [12.6, 13.8, 15.0, 16.2, 17.4, 18.6, 19.8, 21.0]
at_p_list = [6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6]
damage1_list = [6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6]
damage2_list = [6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6]
efficiency_list = [6.8, 7.6, 8.4, 9.2, 10.0, 10.8, 11.6, 12.4]
at_n_list = [30, 40, 50, 60, 0, 0, 0, 0]

# 出現確率ウェイト
cr_weight = [0.2429, 0.2458, 0.2023, 0.0940, 0.0840, 0.0872, 0.0269, 0.0169]
cd_weight = [0.157509, 0.274725, 0.282051, 0.076923, 0.091575, 0.080586, 0.021978, 0.014652]
at_p_weight = [0.080862, 0.086253, 0.185983, 0.226415, 0.153638, 0.194070, 0.051212, 0.021563]
damage1_weight = at_p_weight
damage2_weight = at_p_weight
efficiency_weight = at_p_weight
at_n_weight = [0.269, 0.385, 0.325, 0.021, 0, 0, 0, 0]

# 統合リスト
subst_list = [cr_list, cd_list, at_p_list, damage1_list, damage2_list, efficiency_list, at_n_list]
subst_weight = [cr_weight, cd_weight, at_p_weight, damage1_weight, damage2_weight, efficiency_weight, at_n_weight]

# コスト定義
record_list = [1100, 3025, 5775, 9875, 15875] 
TUNER_COST_PER_SLOT = 10 
INF = 10**18


# キャラプリセット定義
CHAR_PRESETS = {
    "カスタム (手動設定)": { 
        "coe":[2.0, 1.0, 1.0, 0.7, 0.0, 1.0, 0.1],
        "d1_label": "ダメアップ1",
        "d2_label": "ダメアップ2"
        },
    "リンネー": {
        "coe":[2.0, 1.0, 1.0, 0.6, 0.0, 1.0, 0.1],
        "d1_label": "通常ダメ",
        "d2_label": "ーーーー"
        },
    "千咲": {
        "coe":[2.0, 1.0, 1.0, 0.8, 0.0, 1.0, 0.1],
        "d1_label": "解放ダメ",
        "d2_label": "ーーーー"},
    "仇遠": {
        "coe":[2.0, 1.0, 1.0, 0.6, 0.0, 1.0, 0.1],
        "d1_label": "重撃ダメ",
        "d2_label": "ーーーー"
        },
    "ガルブレーナ": {
        "coe":[2.0, 1.0, 1.0, 0.3, 0.0, 1.0, 0.1],
        "d1_label": "重撃ダメ",
        "d2_label": "ーーーー"
        },
    "ユーノ": {
        "coe":[2.0, 1.0, 1.0, 0.8, 0.0, 1.0, 0.1],
        "d1_label": "解放ダメ",
        "d2_label": "ーーーー"
        },
    "オーガスタ":{
        "coe": [2.0, 1.0, 1.0, 0.8, 0.0, 1.0, 0.1],
        "d1_label": "重撃ダメ",
        "d2_label": "ーーーー"
        },
    "フローヴァ":{ 
        "coe":[2.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.1],
        "d1_label": "スキルダメ",
        "d2_label": "ーーーー"
        },
    "ルパ": {
        "coe":[2.0, 1.0, 1.0, 0.7, 0.0, 1.0, 0.1],
        "d1_label": "解放ダメ",
        "d2_label": "ーーーー"
        },
    "シャコンヌ": {
        "coe":[2.0, 1.0, 1.0, 0.3, 0.0, 1.0, 0.1],
        "d1_label": "解放ダメ",
        "d2_label": "ーーーー"
        },
    "ザンニー": {
        "coe":[2.0, 1.0, 1.0, 0.7, 0.0, 1.0, 0.1],
        "d1_label": "重撃ダメ",
        "d2_label": "ーーーー"
        },
    "カンタレラ": {
        "coe":[2.0, 1.0, 1.0, 0.7, 0.0, 1.0, 0.1],
        "d1_label": "通常ダメ",
        "d2_label": "ーーーー"},
    "ブラント": {
        "coe":[2.0, 1.0, 1.0, 0.7, 0.0, 1.0, 0.1],
        "d1_label": "通常ダメ",
        "d2_label": "ーーーー"
        },
    "フィービー": {
        "coe":[2.0, 1.0, 1.0, 0.4, 0.0, 1.0, 0.1],
        "d1_label": "重撃ダメ",
        "d2_label": "ーーーー"
        },
    "ロココ": {
        "coe":[2.0, 1.0, 1.0, 0.6, 0.0, 1.0, 0.1],
        "d1_label": "重撃ダメ",
        "d2_label": "ーーーー"},
    "カルロッタ": {
        "coe":[2.0, 1.0, 1.0, 0.8, 0.0, 1.0, 0.1],
        "d1_label": "スキルダメ",
        "d2_label": "ーーーー"
        },
    "ツバキ": {
        "coe":[2.0, 1.0, 1.0, 0.6, 0.0, 1.0, 0.1],
        "d1_label": "通常ダメ",
        "d2_label": "ーーーー"},
    "折枝": {
        "coe":[2.0, 1.0, 1.0, 0.8, 0.0, 1.0, 0.1],
        "d1_label": "通常ダメ",
        "d2_label": "ーーーー"
        },
    "相里要": {
        "coe":[2.0, 1.0, 1.0, 0.6, 0.0, 1.0, 0.1],
        "d1_label": "解放ダメ",
        "d2_label": "ーーーー"
        },
    "長離": {
        "coe":[2.0, 1.0, 1.0, 0.6, 0.0, 1.0, 0.1],
        "d1_label": "スキルダメ",
        "d2_label": "ーーーー"},
    "今汐": {
        "coe":[2.0, 1.0, 1.0, 0.8, 0.0, 0.0, 0.1],
        "d1_label": "スキルダメ",
        "d2_label": "ーーーー"
        },
    "吟霖": {
        "coe":[2.0, 1.0, 1.0, 0.5, 0.0, 1.0, 0.1],
        "d1_label": "スキルダメ",
        "d2_label": "ーーーー"
        },
    "忌炎": {
        "coe":[2.0, 1.0, 1.0, 0.7, 0.0, 1.0, 0.1],
        "d1_label": "重撃ダメ",
        "d2_label": "ーーーー"
        },
    "気道主人公": {
        "coe":[2.0, 1.0, 1.0, 0.7, 0.0, 1.0, 0.1],
        "d1_label": "スキルダメ",
        "d2_label": "ーーーー"
        },
    "消滅主人公": {
        "coe":[2.0, 1.0, 1.0, 0.3, 0.3, 1.0, 0.1],
        "d1_label": "通常ダメ",
        "d2_label": "解放ダメ"
        },
    "回折主人公": {
        "coe":[2.0, 1.0, 1.0, 0.4, 0.3, 1.0, 0.1],
        "d1_label": "スキルダメ",
        "d2_label": "解放ダメ"
        },
    "アンコ": {
        "coe":[2.0, 1.0, 1.0, 0.5, 0.0, 1.0, 0.1],
        "d1_label": "通常ダメ",
        "d2_label": "ーーーー"
        },
    "カカロ": {
        "coe":[2.0, 1.0, 1.0, 0.5, 0.0, 1.0, 0.1],
        "d1_label": "解放ダメ",
        "d2_label": "ーーーー"
        },
    "凌陽": {
        "coe":[2.0, 1.0, 1.0, 0.3, 0.3, 1.0, 0.1],
        "d1_label": "通常ダメ",
        "d2_label": "スキルダメ"
        },
    "灯灯": {
        "coe":[2.0, 1.0, 1.0, 0.4, 0.0, 1.0, 0.1],
        "d1_label": "通常ダメ",
        "d2_label": "ーーーー"
        },
    "アールト": {
        "coe":[2.0, 1.0, 1.0, 0.4, 0.0, 1.0, 0.1],
        "d1_label": "通常ダメ",
        "d2_label": "ーーーー"},
    "熾霞": {
        "coe":[2.0, 1.0, 1.0, 0.5, 0.3, 1.0, 0.1],
        "d1_label": "スキルダメ",
        "d2_label": "解放ダメ"
        },
    "モルトフィー": {
        "coe":[2.0, 1.0, 1.0, 0.6, 0.0, 1.0, 0.1],
        "d1_label": "解放ダメ",
        "d2_label": "ーーーー"
        },
    "丹瑾": {
        "coe":[2.0, 1.0, 1.0, 0.4, 0.0, 1.0, 0.1],
        "d1_label": "重撃ダメ",
        "d2_label": "ーーーー"
        },
    "散華": {
        "coe":[2.0, 1.0, 1.0, 0.3, 0.3, 1.0, 0.1],
        "d1_label": "解放ダメ",
        "d2_label": "スキルダメ"
        },
}

# --- 計算ロジック ---
def cal_score_now(substatus,coe):#現在スコア
    s=0
    for i in range(7):
        s+=substatus[i]*coe[i]
    return s

def possible_next_states(substatus, t, coe):
    """
    サブステ13種対応・正確確率モデル
    (next_substatus, probability) を列挙
    """
    next_states = []

    # 未取得サブステの数
    empty_idx = [i for i, v in enumerate(substatus) if v == 0]
    n = 8 + t
    #不要サブステの内の未取得サブステの数
    m = 8 - len(empty_idx) + t
    
    p_choose_type = 1 / n  # 種類抽選は等確率
    
    #不要サブステ（一括で処理）
    sub_next = list(substatus)
    next_states.append(
        (tuple(sub_next),p_choose_type * m)
    )

    for i in empty_idx:
        # 有効サブステ（スコアに寄与しない）
        if coe[i] == 0:
            sub_next = list(substatus)
            sub_next[i] = 1 #スコアはどうせ0なので0以外の適当な数字を入れて抽選済みなことを示す
            next_states.append(
                (tuple(sub_next), p_choose_type)
            )
            continue

        # 有効サブステ（スコアに寄与）
        for val, w in zip(subst_list[i], subst_weight[i]):
            sub_next = list(substatus)
            sub_next[i] = val
            next_states.append(
                (tuple(sub_next), p_choose_type * w)
            )

    return next_states
@lru_cache(None)
def expected_chuna(substatus, t, target_score, ave_chuna, coe):
    """
    状態 (substatus, t) から目標スコアに到達するための
    期待チュナ消費量を返す
    """
    score_now = cal_score_now(substatus, coe)

    # 成功
    if score_now >= target_score:
        return 0, 1, True

    # もう強化できない
    if t == 0:
        return INF, 0, False

    total_prob = 0.0
    total_cost = 0.0

    for sub_next, prob in possible_next_states(substatus, t, coe):
        E_next, prob_success, ok = expected_chuna(
            sub_next,
            t - 1,
            target_score,
            ave_chuna,
            coe
        )

        if not ok:
            continue

        if E_next > ave_chuna:
            continue  # 枝刈り（今のロジックと同じ）

        total_prob += prob * prob_success
        total_cost += prob * prob_success * E_next

    if total_prob == 0:
        return INF, 0, False

    # 強化1回分の固定コストを加算
    expected = (7 + total_cost) / total_prob
    
    return expected, total_prob, True

def cal_ave_chuna_fast(target_score,times,substatus,ave_chuna,coe):
    substatus = tuple(substatus)
    coe = tuple(coe)
    return expected_chuna(substatus,5-times,target_score,ave_chuna,coe)[0] + 3 * (5-times)

def number_effective_subst(coe):
    n=0
    for i in range(7):
        if coe[i]>0:
            n+=1
    return n       

def cal_ave_chuna4(score,substatus,coe):
    chuna=0
    record=0
    prob=0
    score_now=cal_score_now(substatus,coe)
    if score_now>=score:
        return 10,1,record_list[4]*4
    for i in range(7):
        if coe[i]==0:
            continue
        if substatus[i]>0:
            continue
        for k in range(8):
            if coe[i]*subst_list[i][k]+score_now>=score:
                prob+=subst_weight[i][k]
    prob=prob/9
    if prob==0:
        return chuna,prob,record
    chuna=7/prob+3
    record=record_list[4]/prob+record_list[4]*3
    return chuna,prob,record

def cal_effective_subst(substatus):#有効サブステの数
    n=0
    for i in range(7):
        if substatus[i]>0:
            n+=1
    return n           
            

def cal_ave_chuna3(score,substatus,ave_chuna,coe):
    chuna=0
    record=0
    prob1=0 #1回だけ強化
    prob2=0 #2回とも強化して成功
    score_now=cal_score_now(substatus,coe)
    
    if score_now>=score:
        record=(record_list[3]+record_list[4])*4
        return 20,1,record
    for i in range(7):
        if coe[i]==0:
            continue
        if substatus[i]>0:
            continue
        for j in range(8):
            substatus_next=copy.copy(substatus)
            substatus_next[i]=subst_list[i][j]
            cal_chuna4=cal_ave_chuna4(score,substatus_next,coe)
            if cal_chuna4[1]>0 and cal_chuna4[0]<=ave_chuna:
                prob2+=subst_weight[i][j]*cal_chuna4[1]
            else:
                prob1+=subst_weight[i][j]
    
    prob1=prob1/10
    prob2=prob2/10
    cal_chuna4=cal_ave_chuna4(score,substatus,coe)
    
    if cal_chuna4[1]>0 and cal_chuna4[0]<=ave_chuna:
        prob2+=cal_chuna4[1]*(10-number_effective_subst(coe)+cal_effective_subst(substatus))/10
    else:
        prob1+=(10-number_effective_subst(coe)+cal_effective_subst(substatus))/10
        
    if prob2==0:
        return chuna,prob2,record
    
    chuna=(14-7*prob1)/prob2+6
    record=(record_list[4]+record_list[3]-record_list[4]*prob1)/prob2+(record_list[3]+record_list[4])*3
    return chuna,prob2,record      
                
def cal_ave_chuna2(score,substatus,ave_chuna,coe):
    chuna=0
    record=0
    chuna_1time_1=0 #3回目に有効ステが着いたときの消費チュナ期待値
    chuna_1time_2=0 #3回目に不要ステが着いたときの消費チュナ期待値
    record_1time_1=0
    record_1time_2=0
    prob=0 #3回とも強化して成功
    score_now=cal_score_now(substatus,coe)
    
    if score_now>=score:
        record=(record_list[2]+record_list[3]+record_list[4])*4
        return 30,1,record
    for i in range(7):
        if coe[i]==0:
            continue
        if substatus[i]>0:
            continue
        for j in range(8):
            substatus_next=copy.copy(substatus)
            substatus_next[i]=subst_list[i][j]
            cal_chuna3=cal_ave_chuna3(score,substatus_next,ave_chuna,coe)
            if cal_chuna3[1]>0 and cal_chuna3[0]<=ave_chuna:
                chuna_1time_1+=subst_weight[i][j]*(cal_chuna3[0]-6)*cal_chuna3[1]
                record_1time_1+=subst_weight[i][j]*(cal_chuna3[2]-(record_list[3]+record_list[4])*3)*cal_chuna3[1]
                prob+=subst_weight[i][j]*cal_chuna3[1]
    
    cal_chuna3=cal_ave_chuna3(score,substatus,ave_chuna,coe)
    
    if cal_chuna3[1]>0 and cal_chuna3[0]<=ave_chuna:
        x=cal_chuna3[1]*(11-number_effective_subst(coe)+cal_effective_subst(substatus))
        prob+=x
        chuna_1time_2+=x*(cal_chuna3[0]-6)
        record_1time_2+=x*(cal_chuna3[2]-(record_list[3]+record_list[4])*3)
        
    if prob==0:
        return chuna,prob,record
    
    chuna=(chuna_1time_1+chuna_1time_2+7*11)/prob+9
    record=(record_1time_1+record_1time_2+record_list[2]*11)/prob+(record_list[2]+record_list[3]+record_list[4])*3
    return chuna,prob/11,record 

def cal_ave_chuna1(score,substatus,ave_chuna,coe):
    chuna=0
    record=0
    chuna_1time_1=0 #2回目に有効ステが着いたときの消費チュナ期待値
    chuna_1time_2=0 #2回目に不要ステが着いたときの消費チュナ期待値
    record_1time_1=0
    record_1time_2=0
    prob=0 #4回とも強化して成功
    score_now=cal_score_now(substatus,coe)
    
    if score_now>=score:
        record=(record_list[1]+record_list[2]+record_list[3]+record_list[4])*4
        return 40,1,record
    for i in range(7):
        if coe[i]==0:
            continue
        if substatus[i]>0:
            continue
        for j in range(8):
            substatus_next=copy.copy(substatus)
            substatus_next[i]=subst_list[i][j]
            cal_chuna2=cal_ave_chuna2(score,substatus_next,ave_chuna,coe)
            if cal_chuna2[1]>0 and cal_chuna2[0]<=ave_chuna:
                chuna_1time_1+=subst_weight[i][j]*(cal_chuna2[0]-9)*cal_chuna2[1]
                record_1time_1+=subst_weight[i][j]*(cal_chuna2[2]-(record_list[2]+record_list[3]+record_list[4])*3)*cal_chuna2[1]
                prob+=subst_weight[i][j]*cal_chuna2[1]

    cal_chuna2=cal_ave_chuna2(score,substatus,ave_chuna,coe)
    
    if cal_chuna2[1]>0 and cal_chuna2[0]<=ave_chuna:
        x=cal_chuna2[1]*(12-number_effective_subst(coe)+cal_effective_subst(substatus))
        prob+=x
        chuna_1time_2+=x*(cal_chuna2[0]-9)
        record_1time_2+=x*(cal_chuna2[2]-(record_list[2]+record_list[3]+record_list[4])*3)
        
    if prob==0:
        return chuna,prob,record
    
    chuna=(chuna_1time_1+chuna_1time_2+7*12)/prob+12
    record=(record_1time_1+record_1time_2+record_list[1]*12)/prob+(record_list[1]+record_list[2]+record_list[3]+record_list[4])*3
    return chuna,prob/12,record  

def cal_ave_chuna0(score,ave_chuna,coe):
    substatus=[0,0,0,0,0,0,0]
    chuna_1time_1=0 #1回目に有効ステが着いて強化継続したときのチュナ消費期待値
    chuna_1time_2=0 #1回目に不要ステが着いて強化継続したときのチュナ消費期待値
    record_1time_1=0
    record_1time_2=0
    prob=0 #最後まで強化して成功
    for i in range(7):
        if coe[i]==0:
            continue
        for j in range(8):
            substatus_next=copy.copy(substatus)
            substatus_next[i]=subst_list[i][j]
            cal_chuna1=cal_ave_chuna1(score,substatus_next,ave_chuna,coe)
            if cal_chuna1[1]>0 and cal_chuna1[0]<=ave_chuna:
                chuna_1time_1+=subst_weight[i][j]*(cal_chuna1[0]-12)*cal_chuna1[1]
                record_1time_1+=subst_weight[i][j]*(cal_chuna1[2]-(record_list[1]+record_list[2]+record_list[3]+record_list[4])*3)*cal_chuna1[1]
                prob+=subst_weight[i][j]*cal_chuna1[1]
                
    cal_chuna1=cal_ave_chuna2(score,substatus,ave_chuna,coe)
    if cal_chuna1[1]>0 and cal_chuna1[0]<=ave_chuna:
        chuna_1time_2+=(13-number_effective_subst(coe))*(cal_chuna1[0]-12)*cal_chuna1[1]
        record_1time_2+=(13-number_effective_subst(coe))*(cal_chuna1[2]-(record_list[1]+record_list[2]+record_list[3]+record_list[4])*3)*cal_chuna1[1]
        prob+=(13-number_effective_subst(coe))*cal_chuna1[1]
    if prob==0:
        return 0,0,0
    chuna=(chuna_1time_1+chuna_1time_2+7*13)/prob+15
    record=(record_1time_1+record_1time_2+record_list[0]*13)/prob+(record_list[0]+record_list[1]+record_list[2]+record_list[3]+record_list[4])*3
    return chuna,prob/13,record

def cal_max_score(coe):#取りうるスコアの最大値を計算
    arr = [coe[i] * subst_list[i][7] for i in range(3, 7)]
    largest, second_largest = sorted(arr, reverse=True)[:2]
    max_score=53.6+largest+second_largest 
    
    return max_score
      
def cal_min_chuna(score,coe):#入力スコア以上の音骸を一つ作るのに必要なチュナ
    chuna1=4000
    chuna2=0
    while abs(chuna1-chuna2)>=1:
        a=cal_ave_chuna_fast(score,0,[0,0,0,0,0,0,0],chuna1,coe)
        if a==0:
            chuna2=300000
            chuna1=cal_ave_chuna_fast(score,0,[0,0,0,0,0,0,0],300000,coe)
        else:    
            chuna2=chuna1
            chuna1=a
    return chuna1    

def judge_continue_all(score,times,ave_chuna,coe):
    results = []
    a=1
    subst_list0=copy.deepcopy(subst_list)
    
    for i in range(7):
        subst_list0[i].insert(0,0)
    memory=np.zeros((1,7))+8
    
    substatus=[0]*7
    chuna=cal_ave_chuna_fast(score,times,substatus,ave_chuna,coe)
    
    if chuna<=ave_chuna*a:
        results.append({
            "substatus": substatus.copy(),
            "chuna": chuna,
            "score": cal_score_now(substatus,coe)
        })
        return results
        

    for i in range(7):
        if coe[i]==0:
            continue
        for j in range(8):
            if i==6 and j>=4:
                break
            substatus=[0]*7
            substatus[i]=subst_list0[i][j+1]
            chuna=cal_ave_chuna_fast(score,times,substatus,ave_chuna,coe)

            if chuna<=ave_chuna*a:
                results.append({
                    "substatus": substatus.copy(),
                    "chuna": chuna,
                    "score": cal_score_now(substatus,coe)
                })

                memory=np.append(memory,[[0]*7],axis=0)
                memory[-1,i]=j+1
                break
            
    if times==1:
        return results
    
    for i in range(6):
        if coe[i]==0:#有効なサブステでない場合パス
            continue

        for j in range(8):
            subst=np.zeros(7)
            subst[i]=j+1
            if np.count_nonzero(np.all(memory<=subst,axis=1))>0:#memory以上の強さのリストを除外
                break

            for k in range(i+1,7):#有効なサブステでない場合パス
                if coe[k]==0:
                    continue
                
                for l in range(8):
                    if k==6 and l>=4:#実数の5番目以降を除外
                        break
                    
                    subst[k]=l+1
                    
                    if np.count_nonzero(np.all(memory<=subst,axis=1))>0:#memory以上の強さのリストを除外
                        subst[k]=0
                        break
                    
                    subst[k]=0
                    
                    substatus=[0]*7
                    substatus[i]=subst_list0[i][j+1]
                    substatus[k]=subst_list0[k][l+1]
                    chuna=cal_ave_chuna_fast(score,times,substatus,ave_chuna,coe)
                    
                    if chuna<=ave_chuna*a:
                        results.append({
                            "substatus": substatus.copy(),
                            "chuna": chuna,
                            "score": cal_score_now(substatus,coe)
                        })

                        memory=np.append(memory,[[0]*7],axis=0)
                        memory[-1,i]=j+1
                        memory[-1,k]=l+1
                        break
    
    if times==2:
        return results
    
    for i in range(5):
        if coe[i]==0:#有効なサブステでない場合パス
            continue

        for j in range(8):
            subst=np.zeros(7)
            subst[i]=j+1
            if np.count_nonzero(np.all(memory<=subst,axis=1))>0:#memory以上の強さのリストを除外
                break

            for k in range(i+1,6):#有効なサブステでない場合パス
                if coe[k]==0:
                    continue
                
                for l in range(8):
                    if k==6 and l>=4:#実数の5番目以降を除外
                        break
                    
                    subst[k]=l+1
                    
                    if np.count_nonzero(np.all(memory<=subst,axis=1))>0:#memory以上の強さのリストを除外
                        subst[k]=0
                        break
                    
                    for m in range(k+1,7):
                        if coe[m]==0:
                            continue
                        
                        for n in range(8):
                            if m==6 and n>=4:
                                break
                            
                            subst[m]=n+1
                            
                            if np.count_nonzero(np.all(memory<=subst,axis=1))>0:
                                subst[m]=0
                                break
                            
                            subst[m]=0
                            
                            substatus=[0]*7
                            substatus[i]=subst_list0[i][j+1]
                            substatus[k]=subst_list0[k][l+1]
                            substatus[m]=subst_list0[m][n+1]
                            chuna=cal_ave_chuna_fast(score,times,substatus,ave_chuna,coe)
                            
                            if chuna<=ave_chuna*a:
                                results.append({
                                    "substatus": substatus.copy(),
                                    "chuna": chuna,
                                    "score": cal_score_now(substatus,coe)
                                })

                                memory=np.append(memory,[[0]*7],axis=0)
                                memory[-1,i]=j+1
                                memory[-1,k]=l+1
                                memory[-1,m]=n+1
                                break
                    
                    subst[k]=0
                    
    if times==3:
        return results
    
    for i in range(4):
        if coe[i]==0:#有効なサブステでない場合パス
            continue

        for j in range(8):
            subst=np.zeros(7)
            subst[i]=j+1
            if np.count_nonzero(np.all(memory<=subst,axis=1))>0:#memory以上の強さのリストを除外
                break

            for k in range(i+1,5):#有効なサブステでない場合パス
                if coe[k]==0:
                    continue
                
                for l in range(8):
                    subst[k]=l+1
                    
                    if np.count_nonzero(np.all(memory<=subst,axis=1))>0:#memory以上の強さのリストを除外
                        subst[k]=0
                        break
                    
                    for m in range(k+1,6):
                        if coe[m]==0:
                            continue
                        
                        for n in range(8):
                            subst[m]=n+1
                            
                            if np.count_nonzero(np.all(memory<=subst,axis=1))>0:
                                subst[m]=0
                                break
                            
                            for o in range(m+1,7):
                                if coe[o]==0:
                                    continue
                                
                                for p in range(8):
                                    if o==6 and p>=4:
                                        break
                                    
                                    subst[o]=p+1
                                    
                                    if np.count_nonzero(np.all(memory<=subst,axis=1))>0:
                                        subst[o]=0
                                        break
                                    
                                    subst[o]=0
                            
                                    substatus=[0]*7
                                    substatus[i]=subst_list0[i][j+1]
                                    substatus[k]=subst_list0[k][l+1]
                                    substatus[m]=subst_list0[m][n+1]
                                    substatus[o]=subst_list0[o][p+1]
                                    chuna=cal_ave_chuna_fast(score,times,substatus,ave_chuna,coe)
                            
                                    if chuna<=ave_chuna*a:
                                        results.append({
                                            "substatus": substatus.copy(),
                                            "chuna": chuna,
                                            "score": cal_score_now(substatus,coe)
                                        })

                                        memory=np.append(memory,[[0]*7],axis=0)
                                        memory[-1,i]=j+1
                                        memory[-1,k]=l+1
                                        memory[-1,m]=n+1
                                        memory[-1,o]=p+1
                                        break
                                    
                            subst[m]=0
                                    
                    subst[k]=0
                    
    if times==4:
        return results   
    
    return "error"
      
def cal_max_score_by_chuna(chuna_limit,coe,score_max, score_min=1):
    lo, hi = score_min, score_max
    best = lo

    while lo <= hi:
        mid = (lo + hi) // 2
        need = cal_min_chuna(mid,coe)

        if need <= chuna_limit:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best

# ==========================================
# 3. Pydanticモデル (リクエスト/レスポンス定義)
# ==========================================

class CalcTargetRequest(BaseModel):
    chuna_limit: float
    coe: List[float] # 係数リスト

class CalcTargetResponse(BaseModel):
    target_score: int
    ave_chuna: float
    res_record: float
    res_body: float

class JudgeRequest(BaseModel):
    target_score: int
    current_times: int
    substatus: List[float]
    ave_chuna: float
    coe: List[float]

class JudgeResponse(BaseModel):
    judgment: str
    expected_cost: float

class LinesRequest(BaseModel):
    target_score: int
    search_times: int
    ave_chuna: float
    coe: List[float]

class LineItem(BaseModel):
    substatus: List[float]
    score: float
    chuna: float

class LinesResponse(BaseModel):
    results: List[LineItem]

# ==========================================
# 4. APIエンドポイント
# ==========================================

@app.get("/")
def read_root():
    return {"message": "Wuthering Waves Echo Calculator API is running"}

@app.post("/api/calc_target", response_model=CalcTargetResponse)
def calc_target(req: CalcTargetRequest):
    """目標スコアと必要リソースを計算"""
    try:
        # 最大スコアは内部計算
        arr = [req.coe[i] * subst_list[i][7] for i in range(len(req.coe)) if req.coe[i] > 0]
        arr.sort(reverse=True)
        max_s = 53.6 + arr[0] + arr[1] if len(arr) >= 2 else 60.0
        
        # 計算
        sc = cal_max_score_by_chuna(req.chuna_limit, req.coe, max_s)
        sc_int = int(sc)
        
        min_c = cal_min_chuna(sc_int, req.coe)
        c, r, prob = cal_ave_chuna0(sc_int, min_c, req.coe)
        
        res_b = 1 / prob if prob > 0 else INF
        
        return {
            "target_score": sc_int,
            "ave_chuna": c,
            "res_record": r,
            "res_body": res_b
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/judge", response_model=JudgeResponse)
def api_judge(req: JudgeRequest):
    """続行判定"""
    substatus_t = tuple(req.substatus)
    coe_t = tuple(req.coe)
    e_tuner, _, ok = expected_chuna(substatus_t, 5-req.current_times, req.target_score, req.ave_chuna, coe_t)
    e_tuner += 3 * (5-req.current_times)
    
    if not ok:
        return {"judgment": "達成不可能", "expected_cost": INF}
    
    if e_tuner <= req.ave_chuna:
        msg = "強化推奨"
    elif e_tuner <= req.ave_chuna * 1.2:
        msg = "続行可能"
    else:
        msg = "強化非推奨"
        
    return {"judgment": msg, "expected_cost": e_tuner}

@app.post("/api/lines", response_model=LinesResponse)
def api_lines(req: LinesRequest):
    """最小ライン一覧取得"""
    raw_results = judge_continue_all(req.target_score, req.search_times, req.ave_chuna, req.coe)
    
    formatted = []
    for r in raw_results:
        formatted.append({
            "substatus": r["substatus"], # list
            "score": r["score"],
            "chuna": r["chuna"]
        })
    
    return {"results": formatted}

# 起動コマンドメモ
# uvicorn backend:app --reload