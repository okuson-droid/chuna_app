import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache
import copy

#チュナ最小になる戦略（新）(レコードの消費量も計算)

cr_list=[6.3, 6.9, 7.5, 8.1, 8.7, 9.3, 9.9, 10.5]
cd_list=[12.6, 13.8, 15.0, 16.2, 17.4, 18.6, 19.8, 21.0]
at_p_list=[6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6]
damage1_list=[6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6]
damage2_list=[6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6]
at_n_list=[30,40,50,60,0,0,0,0]
efficiency_list=[6.8,7.6,8.4,9.2,10,10.8,11.6,12.4]

cr_weight=[0.2429, 0.2458, 0.2023, 0.0940, 0.0840, 0.0872, 0.0269, 0.0169]
cd_weight=[0.1575091575091575, 0.27472527472527475, 0.28205128205128205, 0.07692307692307693, 0.09157509157509157, 0.08058608058608059, 0.02197802197802198,0.014652014652014652]
at_p_weight=[0.08086253369272237, 0.0862533692722372, 0.18598382749326145, 0.22641509433962265, 0.15363881401617252, 0.1940700808625337, 0.05121293800539083, 0.0215633423180593]
damage1_weight=[0.08086253369272237, 0.0862533692722372, 0.18598382749326145, 0.22641509433962265, 0.15363881401617252, 0.1940700808625337, 0.05121293800539083, 0.0215633423180593]
damage2_weight=[0.08086253369272237, 0.0862533692722372, 0.18598382749326145, 0.22641509433962265, 0.15363881401617252, 0.1940700808625337, 0.05121293800539083, 0.0215633423180593]
efficiency_weight=[0.08086253369272237, 0.0862533692722372, 0.18598382749326145, 0.22641509433962265, 0.15363881401617252, 0.1940700808625337, 0.05121293800539083, 0.0215633423180593]
at_n_weight=[0.269,0.385,0.325,0.021,0,0,0,0]

subst_list=[cr_list,cd_list,at_p_list,damage1_list,damage2_list,efficiency_list,at_n_list]
subst_weight=[cr_weight,cd_weight,at_p_weight,damage1_weight,damage2_weight,efficiency_weight,at_n_weight]
record_list=[1100,3025,5775,9875,15875]
INF = 10**18

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

def judge_continue(score,times,substatus,ave_chuna,coe):#強化続行判定
    chuna=0
    if times==0 or times==1 or times==2 or times==3 or times==4:
        chuna=cal_ave_chuna_fast(score,times,substatus,ave_chuna,coe)

    else:
        return -1,"error"
    
    if chuna<=ave_chuna:
        return int(chuna),"強化推奨"
    elif chuna<=ave_chuna*1.2:
        return int(chuna),"続行可能"
    else:
        return int(chuna),"強化非推奨"
   
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

def highlight_positive(val):
    try:
        if float(val) > 0:
            return "font-weight: bold;"
    except:
        pass
    return ""       

def smart_round(x):
    if isinstance(x, float):
        if x.is_integer():
            return int(x)
        return round(x, 2)
    return x

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

def substat_slider(name, value_list, enabled=True, key=None):
    return st.select_slider(
        name,
        options=[0.0] + value_list,
        value=0.0,
        disabled=not enabled,
        format_func=lambda x: "未取得" if x == 0 else f"{x}",
        key=key
    )

def calc_score_breakdown(substatus, coe):
    return [substatus[i] * coe[i] for i in range(len(substatus))]
# =========================
# タイトル
# =========================
st.title("音骸厳選用計算ツール")

# =========================
# キャッシュ付き計算
# =========================
@st.cache_data
def cached_cal_min_chuna(score,coe):
    return cal_min_chuna(score,coe)

# =========================
# 共通パラメータ入力
# =========================

# ==========================================
# 3. Streamlit UI (サイドバー部分の改修)
# ==========================================

# --- プリセットデータの定義 ---
# [クリティカル, クリダメ, 攻撃%, ダメ1, ダメ2, 共鳴効率, 攻撃実数]
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

# --- コールバック関数: プリセット変更時に値を更新する ---
def update_presets():
    # 選択されたキャラ名を取得
    selected = st.session_state["char_selector"]
    vals = CHAR_PRESETS[selected]["coe"]
    
    # Session Stateの値を直接書き換える
    st.session_state["ni_cr"] = vals[0]
    st.session_state["ni_cd"] = vals[1]
    st.session_state["ni_atk"] = vals[2]
    st.session_state["ni_d1"] = vals[3]
    st.session_state["ni_d2"] = vals[4]
    st.session_state["ni_er"] = vals[5]
    st.session_state["ni_flat"] = vals[6]

# --- サイドバー実装 ---
with st.sidebar:
    st.header("スコア計算の設定")
    
    # プリセット選択（on_changeで関数を呼び出す）
    selected_char = st.selectbox(
        "キャラ・プリセット選択",
        options=list(CHAR_PRESETS.keys()),
        key="char_selector",
        on_change=update_presets  # ←これが重要です
    )
    
    # 初回起動時などのために現在の選択データを取得
    setting = CHAR_PRESETS[selected_char]
    
    # --- 初期値の設定 ---
    # まだSession Stateにキーがない場合（初回読み込み時）のみ初期値をセットする
    if "ni_cr" not in st.session_state:
        # 初回はカスタム(あるいは一番上の項目)の値を入れる
        init_vals = setting["coe"]
        st.session_state["ni_cr"] = init_vals[0]
        st.session_state["ni_cd"] = init_vals[1]
        st.session_state["ni_atk"] = init_vals[2]
        st.session_state["ni_d1"] = init_vals[3]
        st.session_state["ni_d2"] = init_vals[4]
        st.session_state["ni_er"] = init_vals[5]
        st.session_state["ni_flat"] = init_vals[6]

    st.caption("各サブステータスのスコア係数")
    
    coe_input = [0.0] * 7
    
    # keyを指定することで、update_presets関数から値を操作できるようにする
    coe_input[0] = st.number_input("クリティカル", step=0.1, format="%.1f", key="ni_cr")
    coe_input[1] = st.number_input("クリダメ", step=0.1, format="%.1f", key="ni_cd")
    coe_input[2] = st.number_input("攻撃力％", step=0.1, format="%.1f", key="ni_atk")
    
    st.divider()
    
    # ラベルは動的、値はkey経由で管理
    coe_input[3] = st.number_input(setting["d1_label"], step=0.1, format="%.1f", key="ni_d1")
    coe_input[4] = st.number_input(setting["d2_label"], step=0.1, format="%.1f", key="ni_d2")
    
    st.divider()
    
    coe_input[5] = st.number_input("共鳴効率", step=0.1, format="%.1f", key="ni_er")
    coe_input[6] = st.number_input("攻撃実数", step=0.01, format="%.2f", key="ni_flat")

    # ロジックに渡す係数リストを更新
    coe = coe_input
    
# サイドバーで選択中の `setting` ("d1_label", "d2_label") を使用してリストを作成
current_sub_names = [
    "クリティカル",
    "クリダメ",
    "攻撃力％",
    setting["d1_label"],  # サイドバーの選択に合わせて変化
    setting["d2_label"],  # サイドバーの選択に合わせて変化
    "共鳴効率",
    "攻撃実数"
]

st.title("🔊 音骸スコア計算")

st.caption("※ 最大強化済み音骸を想定")

st.subheader("サブステ入力")

substatus = [0.0] * 7
active_indices = [i for i in range(7) if coe[i] > 0]
cols = st.columns(3)

for idx, i in enumerate(active_indices):
    with cols[idx % 3]:
        substatus[i] = substat_slider(
            current_sub_names[i],
            subst_list[i],
            enabled=True,
            key=f"step1_substat_{i}"
        )

st.divider()

# ---- 有効サブステ数チェック ----
active_count = sum(1 for v in substatus if v > 0)

if active_count > 5:
    st.error(
        f"有効サブステが {active_count} 個あります。\n"
        "最大5つまでしか入力できません。"
    )
elif active_count == 0:
    st.info("サブステを入力してください。")
else:
    # ---- 計算 ----
    total_score = cal_score_now(substatus, coe)
    breakdown = calc_score_breakdown(substatus, coe)

    st.subheader("計算結果")

    st.metric(
        label="合計スコア",
        value=f"{total_score:.2f}"
    )

    # ---- 内訳表（0は除外） ----
    df = pd.DataFrame({
        "サブステ": sub_names,
        "値": substatus,
        "係数": coe,
        "スコア寄与": breakdown
    })

    df = df[df["値"] > 0]

    st.subheader("③ スコア内訳")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # ---- CSV DL ----
    csv = df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="📥 スコア内訳をCSVでダウンロード",
        data=csv,
        file_name="relic_score.csv",
        mime="text/csv"
    )


# ==========================================
# 4. メインエリア (動的ラベル適用版)
# ==========================================


# --- タブ表示 ---
tab1, tab2, tab3 = st.tabs(["① 目標設定", "② 続行判定", "③ 最小ライン一覧"])

# セッションステート初期化
if 'target_score' not in st.session_state:
    st.session_state['target_score'] = 25.0
if 'ave_chuna' not in st.session_state:
    st.session_state['ave_chuna'] = 100.0

# --- TAB 1: 目標設定 ---
with tab1:
    st.header("目標スコアの算出および、その目標スコア達成のための素材の消費量を表示")
    st.info("自分が許容できるコスト（チュナ量）から、目指すべき現実的なスコアを逆算します。目標スコアが決まっている人は直接入力してください。")
    
    col1, col2 = st.columns(2)
    with col1:
        limit_chuna = st.number_input("使用可能なチュナの上限", min_value=1,value=500, step=100)
    
        if st.button("目標スコアを計算する"):
            with st.spinner("計算中..."):
                max_s = cal_max_score(coe)
                calc_score = cal_max_score_by_chuna(limit_chuna, coe, max_s)
                st.session_state['target_score'] =int(calc_score)
            
                # そのスコアに対する正確な必要コストを再計算
                req_chuna = cal_min_chuna(calc_score, coe)
                c,prob,r = cal_ave_chuna0(calc_score,req_chuna,coe)
                st.session_state['ave_chuna'] = c
                st.session_state['res_c'] = c
                st.session_state['res_r'] = r / 5000
                st.session_state['res_b'] = 1 / prob if prob > 0 else INF
        
            st.success(f"推奨目標スコア: **{calc_score:.0f}**")
            
    with col2:
        val = st.number_input("目標スコアを直接入力", value=st.session_state['target_score'], step=1)
        if st.button("素材の消費量を計算"):
            st.session_state['target_score'] = val
            with st.spinner("計算中..."):
                req_chuna = cal_min_chuna(val, coe)
                c, prob, r= cal_ave_chuna0(val, req_chuna, coe)
                
                st.session_state['ave_chuna'] = c
                st.session_state['res_c'] = c
                st.session_state['res_r'] = r
                st.session_state['res_b'] = 1 / prob if prob > 0 else INF
                st.toast("計算完了")

st.divider()
    
# リソース消費量の表示
if 'res_c' in st.session_state:
    st.subheader("素材消費量（期待値）")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("チュナ消費量", f"{int(st.session_state['res_c']):,}")
    m2.metric("レコード消費量(金レコ換算)", f"{int(st.session_state['res_r']):,}")
        
    b_val = st.session_state['res_b']
    b_str = f"{b_val:.0f} 個" if b_val < 10**10 else "∞"
    m3.metric("音骸素体消費量", b_str)
else:
    st.caption("※計算ボタンを押すとここに消費量が表示されます")


# --- TAB 2: 続行判定 (ラベル修正) ---
with tab2:
    st.header("強化続行・撤退の判定")
    st.markdown(f"目標スコア **{st.session_state['target_score']:.0f}** を目指す場合の判定を行います。")
    
    # 入力フォーム
    col_input, col_res = st.columns([1, 1])
    
    with col_input:
        st.subheader("現在の音骸ステータス")
        
        current_times = st.slider("強化回数（サブステがいくつ開けられているか）", 0, 4, 1)
        
        current_sub = [0.0] * 7
        
        # 有効な（係数が0より大きい）ステータスのみ入力させる
        active_indices = [i for i, c in enumerate(coe) if c > 0]
        
        if len(active_indices) == 0:
            st.warning("サイドバーで係数を1つ以上設定してください。")
        
        input_count = 0
        for i in active_indices:
            # スライダーで選択（リストにある値のみ）
            vals = [0.0] + subst_list[i]
            val = st.select_slider(
                f"{current_sub_names[i]}", # ← ここを動的リストに変更しました
                options=vals,
                key=f"slider_{i}"
            )
            current_sub[i] = val
            if val > 0: input_count += 1
            
        if input_count > current_times:
            st.error(f"入力されたサブステ数({input_count})が強化回数({current_times})を超えています。")

    with col_res:
        st.subheader("判定結果")
        
        current_score = cal_score_now(current_sub, coe)
        st.metric("現在のスコア", f"{current_score:.2f}")
        
        if st.button("判定実行", type="primary"):
            if input_count > current_times:
                st.error("入力値を確認してください")
            else:
                cost, msg = judge_continue(
                    st.session_state['target_score'],
                    current_times,
                    current_sub,
                    st.session_state['ave_chuna'],
                    coe
                )
                
                if msg == "強化推奨":
                    st.success(f"## {msg}")
                    st.markdown("このまま強化を続けるのが期待値的に**お得**です。")
                elif "続行可能" in msg:
                    st.warning(f"## {msg}")
                    st.markdown("レベル0の音骸を強化するのとあまり変わりません。")
                else:
                    st.error(f"## {msg}")
                    st.markdown("これ以上強化すると期待値的に**損**です。次の音骸に行きましょう。")
                
                if cost < INF:
                    st.write(f"ゴールまでの期待コスト: **{cost:.1f}** チュナ")
                    diff = st.session_state['ave_chuna'] - cost
                    if diff > 0:
                        st.write(f"レベル0音骸を育成するより平均 **{diff:.1f}** チュナ分有利です。")
                    else:
                        st.write(f"レベル0音骸を育成するより平均 **{abs(diff):.1f}** チュナ分余計にかかる見込みです。")

# --- TAB 3: 最小ライン一覧 (judge_continue_all 使用) ---
with tab3:
    st.header("これ以上なら強化続行するべき最小ライン一覧")
    st.info("「このサブステが付いたら強化を続けても良い」という最低ラインの組み合わせを表示します。")
    st.caption("※上位互換となる（より強い）組み合わせは、自動的に省略されています。")

    if 'target_score' not in st.session_state:
        st.error("先にタブ①で目標スコアを計算してください")
    else:
        search_times = st.selectbox("検索する強化回数", [1, 2, 3, 4], help="回数が多いと計算に時間がかかります")
        
        if st.button("一覧を生成"):
            with st.spinner("探索中..."):
                ave = st.session_state['ave_chuna']
                score_target = st.session_state['target_score']
                
                # judge_continue_all を呼び出す
                results = judge_continue_all(score_target, search_times, ave, coe)
                
                if not results:
                    st.warning("条件を満たす組み合わせが見つかりませんでした。（すべて非推奨です）")
                else:
                    # 辞書リストをDataFrameに変換
                    # resultsの中身: [{"substatus": [0,0...], "chuna": 123, "score": 20}, ...]
                    
                    rows = []
                    for r in results:
                        row = {}
                        sub = r["substatus"]
                        # カラム名マッピング
                        for i in range(7):
                            row[current_sub_names[i]] = sub[i]
                        row["スコア"] = r["score"]
                        row["消費チュナ"] = r["chuna"]
                        rows.append(row)
                        
                    df = pd.DataFrame(rows)
                    
                    # --- 表示用整形 ---
                    # カラム順序の固定
                    base_order = [
                        "クリティカル", "クリダメ", "攻撃力％",
                        setting["d1_label"], setting["d2_label"],
                        "共鳴効率", "攻撃実数"
                    ]
                    # 係数が0の列は除外
                    active_cols = [name for i, name in enumerate(base_order) if coe[i] > 0]
                    final_cols = active_cols + ["スコア", "消費チュナ"]
                    
                    df_display = df[final_cols]
                    
                    # スコア順にソート
                    df_display = df_display.sort_values("スコア").reset_index(drop=True)
                    
                    # --- スタイルとフォーマット ---
                    # 0の値を目立たなくするスタイル
                    def style_zeros(val):
                        if isinstance(val, (int, float)) and val == 0:
                            return 'color: #d0d0d0; font-weight: 300;'
                        return ''
                    
                    # 数値フォーマットの定義 (すべて小数点1桁)
                    col_config = {}
                    for col in active_cols:
                        col_config[col] = st.column_config.NumberColumn(format="%.1f")
                    col_config["スコア"] = st.column_config.NumberColumn(format="%.1f")
                    col_config["消費チュナ"] = st.column_config.NumberColumn(format="%d")

                    st.write(f"**強化回数 {search_times}回目** の続行可能最小ライン ({len(df_display)}件)")
                    
                    st.dataframe(
                        df_display.style.map(style_zeros),
                        column_config=col_config,
                        use_container_width=True,
                        hide_index=True
                    )







    




