import streamlit as st
import numpy as np
import pandas as pd
import copy
import itertools

#チュナ最小になる戦略（新）(レコードの消費量も計算)

coe=[2,1,1,0,0,0,0.1]

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

sub_names = [
    "クリティカル",
    "クリダメ",
    "攻撃力％",
    "ダメアップ1",
    "ダメアップ2",
    "共鳴効率",
    "攻撃実数"
]

def cal_score_now(substatus,coe):#現在スコア
    s=0
    for i in range(7):
        s+=substatus[i]*coe[i]
    return s


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

def cal_min_chuna(score,coe):#入力スコア以上の音骸を一つ作るのに必要なチュナ
    chuna1=4000
    chuna2=0
    while abs(chuna1-chuna2)>=1:
        chuna2=chuna1
        chuna1=cal_ave_chuna0(score,chuna1,coe)[0]
    return chuna1    

def judge_continue(score,times,substatus,ave_chuna,coe):#強化続行判定
    chuna=0
    if times==1:
        chuna=cal_ave_chuna1(score,substatus,ave_chuna,coe)[0]
    elif times==2:
        chuna=cal_ave_chuna2(score,substatus,ave_chuna,coe)[0]
    elif times==3:
        chuna=cal_ave_chuna3(score,substatus,ave_chuna,coe)[0]
    elif times==4:
        chuna=cal_ave_chuna4(score,substatus,coe)[0]
    else:
        return -1,"error"
    
    if chuna<=ave_chuna:
        return int(chuna),"強化推奨"
    elif chuna<=ave_chuna*1.2:
        return int(chuna),"続行可能"
    else:
        return int(chuna),"強化非推奨"
   
def cal_ave_chuna(score,times,substatus,ave_chuna,coe):
    if times==0:
        return cal_ave_chuna0(score,ave_chuna,coe)
    elif times==1:
        return cal_ave_chuna1(score,substatus,ave_chuna,coe)
    elif times==2:
        return cal_ave_chuna2(score,substatus,ave_chuna,coe)
    elif times==3:
        return cal_ave_chuna3(score,substatus,ave_chuna,coe)
    elif times==4:
        return cal_ave_chuna4(score,substatus,coe)
    else:
        print("error")

def judge_continue_all(score,times,ave_chuna,coe):
    results = []
    a=1
    subst_list0=copy.deepcopy(subst_list)
    
    for i in range(7):
        subst_list0[i].insert(0,0)
    memory=np.zeros((1,7))+8
    
    substatus=[0]*7
    chuna=cal_ave_chuna(score,times,substatus,ave_chuna,coe)
    
    if chuna[1]>0 and chuna[0]<=ave_chuna*a:
        results.append({
            "substatus": substatus.copy(),
            "chuna": chuna[0],
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
            chuna=cal_ave_chuna(score,times,substatus,ave_chuna,coe)

            if chuna[1]==0:
                continue

            if chuna[0]<=ave_chuna*a:
                results.append({
                    "substatus": substatus.copy(),
                    "chuna": chuna[0],
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
                    chuna=cal_ave_chuna(score,times,substatus,ave_chuna,coe)
                    if chuna[1]==0:
                        continue
                    
                    if chuna[0]<=ave_chuna*a:
                        results.append({
                            "substatus": substatus.copy(),
                            "chuna": chuna[0],
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
                            chuna=cal_ave_chuna(score,times,substatus,ave_chuna,coe)
                            if chuna[1]==0:
                                continue
                            
                            if chuna[0]<=ave_chuna*a:
                                results.append({
                                    "substatus": substatus.copy(),
                                    "chuna": chuna[0],
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
                                    chuna=cal_ave_chuna(score,times,substatus,ave_chuna,coe)
                                    if chuna[1]==0:
                                        continue
                            
                                    if chuna[0]<=ave_chuna*a:
                                        results.append({
                                            "substatus": substatus.copy(),
                                            "chuna": chuna[0],
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

def cal_max_score_by_chuna(chuna_limit,coe, score_min=1, score_max=80):
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

def substat_slider(name, values, enabled=True):
    """
    values : そのサブステが取りうる数値リスト（長さ8）
    enabled: 係数0なら False
    """
    idx = st.slider(
        name,
        min_value=0,
        max_value=len(values),
        value=0,
        step=1,
        disabled=not enabled,
    )

    if idx == 0:
        st.caption("未取得")
        return 0.0
    else:
        val = values[idx - 1]
        st.caption(f"選択値：{val}")
        return val

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


with st.sidebar:
    st.header("基本設定")

    st.write("スコア計算に使用するサブステ係数")
    st.caption("※ ここで設定した値は全体の判定・計算に影響します")

    st.markdown("---")

    st.caption("一般的なアタッカー：○○ダメージアップ1の係数 ≒ 0.7")
    st.caption("共鳴効率：サブステで30盛りたいなら 1 / 20なら 0.6")
    st.caption("2種のダメUPが有効なキャラ以外、ダメアップ2は 0 のままでOK")
    st.caption("※ HP参照キャラ（カルテジア）は非対応")

    coe = [2, 1, 1, 0, 0, 0, 0.1]

    coe[3] = st.number_input(
        "○○ダメージアップ1 係数",
        value=0.7,
        step=0.1
    )

    coe[4] = st.number_input(
        "○○ダメージアップ2 係数",
        value=0.0,
        step=0.1
    )

    coe[5] = st.number_input(
        "共鳴効率 係数",
        value=1.0,
        step=0.1
    )


st.header("①使用可能チュナから目標スコアを算出")

chuna_limit = st.number_input(
    "使用可能なチュナ量",
    value=300,
    min_value=1,
    step=100
)

if st.button("①目標スコアを算出"):
    with st.spinner("計算中…"):
        target_score = cal_max_score_by_chuna(chuna_limit,coe)
        chuna, prob, record = cal_ave_chuna0(target_score, chuna_limit,coe)

    st.subheader("算出結果")
    st.metric("現実的な目標スコア", target_score)
    st.metric("想定チュナ消費量", int(chuna))
    st.metric("想定素体消費量", int(1 / prob) if prob > 0 else "∞")
    
    if st.button("このスコアをSTEP2に使う"):
        st.session_state["score"] = target_score
    

st.header("②目標スコア達成に必要な素材量を算出")

score = st.number_input(
    "目標スコア",
    min_value=1,
    step=1,
    value=st.session_state.get("score", 40)
)

if st.button("②計算する"):
    with st.spinner("計算中..."):
        chuna = cached_cal_min_chuna(score,tuple(coe))
        n = list(cal_ave_chuna0(score, chuna,coe))

    n[0] = int(n[0])
    n[2] = int(n[2] / 5000)

    if n[1] > 0:
        n.append(int(1 / n[1]))
    else:
        n.append("無限大")

    # セッションに保存（②で使う）
    st.session_state["ave_chuna"] = n[0]

    st.subheader("計算結果")
    st.metric("チュナ消費量", n[0])
    st.metric("レコード消費量", n[2])
    st.metric("素体消費量", n[3])

st.header("③音骸の強化続行判定")
st.caption("②の計算結果を元に、現在の音骸が続行ラインを超えているかを判定")

with st.expander("現在のサブステータスを入力", expanded=True):

    times_step3 = st.slider(
        "強化回数（現在いくつのサブステが開けられているか）",
        0, 5, 0, 1,
        key="times_step3"
    )

    st.caption("※ 未取得のサブステは 0 のままにしてください")

    substatus = [0.0] * 7
    active_indices = [i for i in range(7) if coe[i] > 0]
    cols = st.columns(3)

    for idx, i in enumerate(active_indices):
        with cols[idx % 3]:
            substatus[i] = substat_slider(
                sub_names[i],
                subst_list[i],
                enabled=True
            )

# STEP3 判定ボタン押下時
if st.button("③判定する"):

    if "ave_chuna" not in st.session_state:
        st.error("先に②の計算を実行してください")
        st.stop()

    opened = sum(1 for v in substatus if v > 0)
    if opened > times_step3:
        st.warning(
            f"開放されているサブステ数（{opened}）が "
            f"強化回数（{times_step3}）を超えています"
        )
        st.stop()

    ave_chuna = st.session_state.get("ave_chuna", 0)

    with st.spinner("判定中..."):
        result_chuna, result_text = judge_continue(score, times_step3, substatus, ave_chuna, coe)

    st.subheader("判定結果")
    st.metric("判定", result_text)

    # 差額計算
    delta_chuna = result_chuna - ave_chuna


    # 色を推奨度に応じて指定
    if delta_chuna <= 0:  # 続行すべき良い状態
        color = "green"
        sign_text = "少ない"
    else:                 # 続行非推奨の状態
        color = "red"
        sign_text = "多い"

    # 列表示
    col1, col2 = st.columns(2)
    col1.metric("想定チュナ消費量", int(result_chuna))
    col2.markdown(
        f"<h4 style='color:{color}'>境界との差: {abs(round(delta_chuna,2))} ({sign_text})</h4>",
        unsafe_allow_html=True
    )

    if delta_chuna <= 0:
        st.caption("この状態から強化を続けた場合、レベル0からの強化よりも期待値的にお得")
    else:
        st.caption("この状態から強化を続けるよりも、レベル0の音骸を強化した方が期待値的にお得") 

st.header("④これ以上なら強化していい最小ライン一覧")
st.caption("②の計算結果をもとに表示")
st.caption("この表に含まれる行のいずれか一つでも完全に下回っていると強化続行非推奨")
st.caption("逆に、表に含まれる行のいずれか一つと同じかそれを上回っていれば強化続行推奨")

times_step4 = st.slider(
    "強化回数",
    0, 5, 0, 1,
    key="times_step4"
)

if st.button("④一覧を表示"):
    if "ave_chuna" not in st.session_state:
        st.error("先に②の計算を実行してください")
        st.stop()

    ave_chuna = st.session_state["ave_chuna"]

    with st.spinner("計算中…"):
        results = judge_continue_all(score, times_step4, ave_chuna,coe)

    if len(results) == 0:
        st.warning("条件を満たすサブステ構成がありません")
    else:
        rows = []
        for r in results:
            row = {}
            for i, name in enumerate(sub_names):
                if coe[i] > 0:          # ← 係数0は表示しない
                    row[name] = r["substatus"][i]
            row["スコア"] = round(r["score"], 2)
            row["消費チュナ"] = round(r["chuna"], 2)
            rows.append(row)

        df = pd.DataFrame(rows)        
        df_display = df.applymap(smart_round)

        styled_df = (
            df.style
                .format("{:.1f}")          
                .applymap(highlight_positive)
        )
        st.dataframe(styled_df, use_container_width=True)
        
import matplotlib.pyplot as plt

st.header("⑤目標スコア別・必要チュナ量一覧")

st.caption(
    "目標スコアごとに、1体完成させるまでに必要な平均チュナ消費量を表示します。\n"
    "線が急に跳ねる部分は、強化戦略が切り替わる判断境界を表しています。"
)

# === 描画範囲の指定 ===
col1, col2, col3 = st.columns(3)

with col1:
    score_min = st.number_input("最小スコア", value=30, step=1)
with col2:
    score_max = st.number_input("最大スコア", value=80, step=1)
with col3:
    score_step = st.number_input("刻み幅", value=1, step=1)

if score_min >= score_max:
    st.error("最小スコアは最大スコアより小さくしてください")
    st.stop()

# === 計算 ===
if st.button("一覧グラフを表示"):
    scores = list(range(score_min, score_max + 1, score_step))

    with st.spinner("計算中...（少し時間がかかります）"):
        chuna_values = [cached_cal_min_chuna(s,tuple(coe)) for s in scores]

    # === グラフ描画 ===
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scores, chuna_values, marker="o")

    ax.set_xlabel("目標スコア")
    ax.set_ylabel("平均チュナ消費量")
    ax.set_title("目標スコアと必要チュナ量の関係")

    ax.grid(True)
    st.pyplot(fig)

    # === テーブル（おまけ・数値確認用） ===
    df = pd.DataFrame({
        "目標スコア": scores,
        "平均チュナ消費量": [int(x) for x in chuna_values]
    })

    st.dataframe(df, use_container_width=True)

        







    




